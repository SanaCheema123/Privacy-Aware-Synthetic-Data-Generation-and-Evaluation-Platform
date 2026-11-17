"""End to end pipeline: load -> schema -> split -> generate -> evaluate -> report -> export."""
from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import deep_update, load_config, logo_path, resolve, set_seed
from .data import detect_schema, load_csv, split_real
from .evaluation import evaluate_fidelity, evaluate_privacy, evaluate_utility, explain_privacy
from .generators import DISPLAY_NAMES, get_generator
from .imbalance import class_distribution, plan_class_counts
from .logging_utils import get_logger
from .reporting import build_html_report
from .reporting import plots

WEIGHTS = {"fidelity_score": 0.35, "utility_score": 0.35, "privacy_score": 0.30}


def _json_safe(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    return str(o)


def run_pipeline(config: dict | str | None = None, overrides: dict | None = None,
                 data: pd.DataFrame | None = None,
                 progress: Callable[[str, float], None] | None = None) -> dict:
    """Run the full platform. Returns a dict with paths, leaderboard and all metrics.

    config     dict, path to YAML, or None for config/config.yaml
    overrides  nested dict applied on top of the config (used by CLI, API and UI)
    data       optional DataFrame (otherwise data.input_csv is read)
    progress   optional callback(message, fraction) for UIs
    """
    cfg = config if isinstance(config, dict) else load_config(config)
    cfg = deep_update(cfg, overrides or {})
    seed = int(cfg["project"]["random_seed"])
    set_seed(seed)
    out = cfg["outputs"]
    log = get_logger(log_dir=resolve(out["logs_dir"]))
    report = progress or (lambda m, f: None)
    run_id = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    fig_dir = resolve(out["figures_dir"]) / run_id
    syn_dir = resolve(out["synthetic_dir"]) / run_id
    rep_dir = resolve(out["reports_dir"]) / run_id
    mod_dir = resolve(out["models_dir"]) / run_id
    logo = logo_path(cfg)
    if logo is None:
        log.warning("Logo not found at %s; reports will be generated without it.", cfg["project"].get("logo_path"))

    # 1. data and schema ------------------------------------------------------
    dcfg = cfg["data"]
    raw = data if data is not None else load_csv(resolve(dcfg["input_csv"]))
    target = dcfg.get("target_column") or None
    schema = detect_schema(raw, target=target, max_categories=int(dcfg["max_categories"]),
                           drop_columns=dcfg.get("drop_columns"))
    rep_dir.mkdir(parents=True, exist_ok=True)
    schema.save(rep_dir / "schema.json")
    log.info("Loaded %d rows, %d columns (%d modelled, dropped: %s)", len(raw), raw.shape[1],
             len(schema.modelled), schema.dropped or "none")
    model_df = schema.to_model_frame(raw)
    if target:
        model_df = model_df[model_df[target].notna()].reset_index(drop=True)
    train, holdout = split_real(model_df, target if schema.task == "classification" else None,
                                float(dcfg["holdout_size"]), seed)
    report("Data loaded and split", 0.05)

    # 2. class plan -------------------------------------------------------------
    icfg = cfg["imbalance"]
    n_rows = int(icfg.get("num_rows") or len(train))
    plan = None
    if schema.task == "classification":
        plan = plan_class_counts(train, target, icfg.get("strategy", "none"), n_rows, icfg.get("custom_ratios"))
        log.info("Real train class distribution:\n%s", class_distribution(train, target).to_string(index=False))
        log.info("Synthetic class plan (%s): %s", icfg.get("strategy"), plan)

    # 3. generators ---------------------------------------------------------------
    enabled = list(cfg["generators"]["enabled"])
    results, synth_frames, losses, trtr_cache, util_tables = {}, {}, {}, {}, {}
    ecfg = cfg["evaluation"]
    for gi, key in enumerate(enabled):
        name = DISPLAY_NAMES.get(key, key)
        base_frac = 0.1 + 0.8 * gi / len(enabled)
        report(f"Training {name}", base_frac)
        log.info("[%s] training ...", name)
        gen = get_generator(key, cfg["generators"].get(key, {}), seed)
        try:
            gen.fit(train, schema)
        except Exception as e:  # keep other generators running
            log.error("[%s] failed: %s", name, e)
            results[key] = {"display_name": name, "error": str(e)}
            continue
        log.info("[%s] trained in %.1f s, sampling ...", name, gen.fit_seconds)
        report(f"Sampling and evaluating {name}", base_frac + 0.4 / len(enabled))
        t0 = time.time()
        synth = gen.sample_conditional(target, plan) if plan else gen.sample(n_rows)
        synth = synth[train.columns]
        notes = []
        shortfall = getattr(gen, "class_shortfall_", {})
        if shortfall:
            notes.append(f"{name} could not generate enough rows for classes {shortfall}; it rarely produces them. "
                         "Train for more epochs or use a generator with native conditioning (TabDDPM).")
            log.warning("[%s] class shortfall %s", name, shortfall)
        if len(synth) < 20:
            log.error("[%s] produced only %d rows, skipping evaluation", name, len(synth))
            results[key] = {"display_name": name, "error": "too few synthetic rows"}
            continue
        sample_s = time.time() - t0
        try:
            gen.save(mod_dir / f"{key}.pkl")
        except Exception as e:
            log.warning("[%s] model not saved: %s", name, e)

        # export in original types
        syn_dir.mkdir(parents=True, exist_ok=True)
        schema.from_model_frame(synth).to_csv(syn_dir / f"synthetic_{key}.csv", index=False)
        synth_frames[name] = synth
        if hasattr(gen, "loss_history"):
            losses[name] = gen.loss_history()

        fid = evaluate_fidelity(train, synth, schema, seed)
        util = evaluate_utility(train, holdout, synth, schema, ecfg["utility_models"], seed, trtr_cache)
        priv = evaluate_privacy(train, holdout, synth, schema, seed, int(ecfg["max_rows_for_privacy"]))
        figs = {
            "distributions": plots.plot_distributions(train, synth, schema, name, fig_dir / f"{key}_distributions.png",
                                                      logo, int(ecfg["figure_max_columns"])),
            "association": plots.plot_association(fid["real_assoc"], fid["synth_assoc"], name,
                                                  fig_dir / f"{key}_associations.png", logo),
            "pca": plots.plot_pca(train, synth, schema, name, fig_dir / f"{key}_pca.png", logo, seed),
            "privacy": plots.plot_privacy(priv, name, fig_dir / f"{key}_privacy.png", logo),
        }
        if util.get("available"):
            util_tables[name] = util["table"]
        results[key] = {"display_name": name, "fit_seconds": gen.fit_seconds, "sample_seconds": sample_s,
                        "n_synthetic": len(synth), "fidelity": fid, "utility": util, "privacy": priv,
                        "privacy_text": explain_privacy(priv), "figures": figs, "notes": notes}
        log.info("[%s] fidelity %.3f | utility %s | privacy %.3f (%s risk)", name, fid["fidelity_score"],
                 f"{util.get('utility_score', float('nan')):.3f}" if util.get("available") else "n/a",
                 priv["privacy_score"], priv["overall_risk"])

    ok = {k: v for k, v in results.items() if "error" not in v}
    if not ok:
        raise RuntimeError(f"All generators failed: { {k: v['error'] for k, v in results.items()} }")

    # 4. leaderboard ---------------------------------------------------------------
    rows = []
    for key, r in ok.items():
        row = {"key": key, "generator": r["display_name"],
               "fidelity_score": r["fidelity"]["fidelity_score"],
               "utility_score": r["utility"].get("utility_score", np.nan),
               "privacy_score": r["privacy"]["privacy_score"],
               "detection_auc": r["fidelity"]["detection_auc"],
               "mia_auc": r["privacy"]["mia_auc"],
               "privacy_risk": r["privacy"]["overall_risk"],
               "train_seconds": r["fit_seconds"]}
        w = {k: v for k, v in WEIGHTS.items() if not pd.isna(row[k])}
        row["overall_score"] = sum(row[k] * v for k, v in w.items()) / sum(w.values())
        rows.append(row)
    board = pd.DataFrame(rows).sort_values("overall_score", ascending=False).reset_index(drop=True)
    board.to_csv(rep_dir / "leaderboard.csv", index=False)

    run_figs = {"leaderboard": plots.plot_leaderboard(board, fig_dir / "leaderboard.png", logo)}
    if plan:
        run_figs["class_balance"] = plots.plot_class_balance(train, synth_frames, target,
                                                             fig_dir / "class_balance.png", logo)
    if util_tables:
        prim = next(iter(ok.values()))["utility"]["primary_metric"]
        run_figs["utility"] = plots.plot_utility(util_tables, prim, fig_dir / "utility_comparison.png", logo)
    lp = plots.plot_losses(losses, fig_dir / "training_losses.png", logo)
    if lp:
        run_figs["losses"] = lp
    report("Building report", 0.95)

    run = {"run_id": run_id, "schema": schema, "n_rows": len(model_df), "n_train": len(train),
           "n_holdout": len(holdout), "imbalance_strategy": icfg.get("strategy", "none"),
           "class_plan": plan, "results": ok, "failed": {k: v["error"] for k, v in results.items() if "error" in v},
           "leaderboard": board, "figures": run_figs}
    html_path = build_html_report(run, rep_dir / "evaluation_report.html", logo)

    # metrics JSON (without big arrays)
    metrics = {"run_id": run_id, "class_plan": plan, "failed": run["failed"],
               "leaderboard": board.to_dict(orient="records"), "generators": {}}
    for key, r in ok.items():
        metrics["generators"][key] = {
            "fidelity": {k: v for k, v in r["fidelity"].items() if k.endswith(("score", "auc"))},
            "utility": {k: v for k, v in r["utility"].items() if k != "table"},
            "utility_table": r["utility"].get("table"),
            "privacy": {k: v for k, v in r["privacy"].items() if not k.startswith("_")},
        }
    (rep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_safe), encoding="utf-8")
    with open(rep_dir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)

    run.update({"report_html": html_path, "synthetic_dir": syn_dir, "figures_dir": fig_dir, "reports_dir": rep_dir,
                "models_dir": mod_dir, "train": train, "holdout": holdout, "synthetic": synth_frames})
    log.info("Run finished. Report: %s", html_path)
    report("Done", 1.0)
    return run
