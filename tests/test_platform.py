"""Unit and smoke tests. Run from the project root: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from make_sample_data import make
from privsynth.data import TabularTransformer, detect_schema, split_real
from privsynth.evaluation import evaluate_fidelity, evaluate_privacy, evaluate_utility
from privsynth.imbalance import plan_class_counts


@pytest.fixture(scope="module")
def data():
    df = make(1200, seed=1)
    schema = detect_schema(df, "outcome")
    mf = schema.to_model_frame(df)
    train, holdout = split_real(mf, "outcome", 0.3, 0)
    return df, schema, train, holdout


def test_schema_detection(data):
    df, schema, *_ = data
    t = {c: i.sdtype for c, i in schema.columns.items()}
    assert t["patient_id"] == "id"
    assert t["admission_date"] == "datetime"
    assert t["age"] == "numerical"
    assert t["sex"] == "boolean"
    assert t["region"] == "categorical"
    assert t["outcome"] == "categorical" and schema.task == "classification"


def test_roundtrip_types(data):
    df, schema, train, _ = data
    back = schema.from_model_frame(train)
    assert back["admission_date"].str.match(r"\d{4}-\d{2}-\d{2}").all()
    assert str(back["age"].dtype) == "Int64"


def test_transformer_roundtrip(data):
    _, schema, train, _ = data
    tf = TabularTransformer(schema.numerical, schema.categorical, mode="quantile").fit(train)
    back = tf.inverse_transform(tf.transform(train))
    assert (back["region"].astype(str) == train["region"].astype(str)).mean() > 0.99
    assert np.allclose(back["age"], train["age"].fillna(train["age"].median()), atol=1.0)


def test_imbalance_plan(data):
    _, _, train, _ = data
    plan = plan_class_counts(train, "outcome", "balance", 1000)
    assert sum(plan.values()) == 1000 and len(set(plan.values())) == 1
    plan = plan_class_counts(train, "outcome", "custom", 500, {"0": 0.7, "1": 0.3})
    assert plan == {"0": 350, "1": 150}


def test_copy_attack_is_detected(data):
    """A generator that copies training rows must be flagged as high risk."""
    _, schema, train, holdout = data
    p = evaluate_privacy(train, holdout, train.copy(), schema)
    assert p["exact_match_rate"] > 0.9
    assert p["mia_auc"] > 0.9
    assert p["overall_risk"] == "high"


def test_independent_sample_is_low_risk(data):
    """Fresh rows from the same distribution should look private."""
    _, schema, train, holdout = data
    fresh = schema.to_model_frame(make(len(train), seed=99))
    p = evaluate_privacy(train, holdout, fresh, schema)
    assert p["exact_match_rate"] < 0.01
    assert abs(p["mia_auc"] - 0.5) < 0.08


def test_fidelity_and_utility(data):
    _, schema, train, holdout = data
    fid = evaluate_fidelity(train, train.sample(frac=1, random_state=0), schema)
    assert fid["column_shape_score"] > 0.99
    util = evaluate_utility(train, holdout, train, schema, ["logistic_regression"])
    assert util["available"] and abs(util["utility_ratio"] - 1) < 1e-9


def test_gaussian_copula_pipeline(tmp_path):
    from privsynth.config import load_config
    from privsynth.pipeline import run_pipeline

    cfg = load_config()
    ov = {"generators": {"enabled": ["gaussian_copula"]},
          "imbalance": {"strategy": "balance", "num_rows": 600},
          "outputs": {k: str(tmp_path / k) for k in cfg["outputs"]}}
    run = run_pipeline(cfg, ov, data=make(800, seed=3))
    syn = pd.read_csv(run["synthetic_dir"] / "synthetic_gaussian_copula.csv")
    assert len(syn) == 600
    assert syn["outcome"].value_counts().nunique() == 1  # balanced
    assert run["report_html"].exists()
