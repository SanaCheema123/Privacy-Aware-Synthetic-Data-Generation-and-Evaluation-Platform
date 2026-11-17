# PrivSynth architecture

![AIVONEX](../assets/aivonex_logo.png)

## Data flow

```
CSV upload
   │
   ▼
load_csv ──► detect_schema ──► Schema.to_model_frame
                                   │  (drops IDs / free text, dates to float days)
                                   ▼
                           split_real (stratified)
                         ┌─────────┴─────────┐
                     TRAIN (members)     HOLDOUT (non members, never seen)
                         │                   │
             plan_class_counts               │
                         │                   │
        ┌────────────────┼───────────────┐   │
   Gaussian Copula   CTGAN   TVAE   TabDDPM  │
        └────────────────┬───────────────┘   │
                         ▼                   │
                 synthetic dataset           │
                         │                   │
        ┌────────────────┼───────────────────┤
   Fidelity        Utility (TSTR vs TRTR)  Privacy (DCR, NN disclosure, MIA)
        └────────────────┬───────────────────┘
                         ▼
     leaderboard.csv · metrics.json · evaluation_report.html · figures/*.png · synthetic_*.csv
```

## Why a holdout split matters

Privacy attacks need people the generator did NOT see. The holdout gives that
reference: if synthetic rows sit closer to training people than to holdout people,
or an attacker can tell members from non members, the generator is leaking.
The same holdout is the real test set for downstream models, so TSTR and TRTR are
measured on identical data.

## Generators

| Key | Model | Library | Imbalance handling |
|-----|-------|---------|--------------------|
| gaussian_copula | Gaussian Copula (statistical baseline) | SDV | conditional sampling |
| ctgan | Conditional GAN with mode specific normalisation | SDV | conditional sampling |
| tvae | Tabular variational autoencoder | SDV | conditional sampling |
| tabddpm | Diffusion model, class conditional MLP denoiser | own PyTorch code | native conditioning |

Adding a new generator: subclass `BaseGenerator` in `src/privsynth/generators/`,
implement `_fit` and `sample`, then register it in `generators/__init__.py`.
Everything else (evaluation, report, UI, API) picks it up automatically.

## Scores

* fidelity_score = mean(column shape score, pair trend score, 1 − 2·|detection AUC − 0.5|)
* utility_score  = clip(TSTR macro F1 / TRTR macro F1, 0, 1)   (R² for regression)
* privacy_score  = 1 − mean of four normalised risks (exact match, close records, NN disclosure, MIA)
* overall_score  = 0.35·fidelity + 0.35·utility + 0.30·privacy

Privacy metrics are empirical risk estimates, not formal differential privacy guarantees.
