# scikit-learn 1.5.2 reference release migration

Issue: [#90](https://github.com/tarekmasryo/fraud-risk-ops-platform/issues/90).

## Provenance and immutable input

The owner supplied a local ZIP. Its declared dataset reference was not accepted on statistics alone. The official Kaggle HTTPS API was queried independently for `mlg-ulb/creditcardfraud`, version 3, dataset ID 310. An independent streamed download through the official versioned API matched the local ZIP byte count and SHA-256 exactly; the final official object was `/kaggle-data-sets/310/23498/bundle/archive.zip` on `storage.googleapis.com`. Signed URL queries were not recorded. Local download-origin metadata was supplemental evidence only.

- Official metadata: https://www.kaggle.com/api/v1/datasets/view/mlg-ulb/creditcardfraud
- Official versioned content: https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud?datasetVersionNumber=3
- ZIP bytes: 69155672; SHA-256: `a0360ce715992212e9ac72d8ccdca97f4be87dc1fdf2bed011358f7ab409a28a`.
- CSV bytes: 150828752; SHA-256: `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
- Sole entry: `creditcard.csv`; no unsafe paths, symlinks, encryption, unexpected files, CRC errors, oversized entries or suspicious compression ratio.
- Exact schema: `Time,V1,...,V28,Amount,Class` (31 columns in that order). All values finite numeric; binary labels only.
- 284807 records: 284315 non-fraud, 492 fraud (0.17274856%). No rows modified, deduplicated, exported as fixtures or committed. The original ZIP was unchanged after validation.

## Reproduction and release identity

Candidate Python 3.11.9, scikit-learn 1.5.2, XGBoost 3.0.2, NumPy 1.26.4, SciPy 1.11.4, joblib 1.4.2 and pandas 2.2.3. Only the scikit-learn pin changes. The original models are not loaded under the new runtime.

Training uses the unchanged `scripts/train.py` at baseline `3a2aba005ac88d686d2b4f51881530d351edc1a9`, directly reading the single-CSV ZIP with pandas. RF uses 300 trees; XGB uses 600 trees and the existing parameters. No automatic MLflow tracking was enabled and existing experiment stores were untouched.

```powershell
python scripts/train.py --data <verified-archive.zip> --out <new-staging-directory> --seed 42 --calibration-size 0.15 --test-size 0.15 --cost-fp 5 --cost-fn 500 --no-tracking
```

Use a new directory; never train directly into the packaged artifacts. The seven files were promoted only after independent evaluation. Release and policy identities are `fraud-risk-ops-v0.1.0-sklearn1.5.2`. Post-training JSON metadata adds the provenance, runtime, split fingerprints and independently computed metrics reported here. Model binaries are direct training outputs. Seeds and split hashes establish repeatability; bitwise equivalence across operating systems/BLAS implementations has not been established.

| Split | Rows | Non-fraud | Fraud |
|---|---:|---:|---:|
| train | 199364 | 199020 | 344 |
| calibration | 42721 | 42647 | 74 |
| holdout | 42722 | 42648 | 74 |

Splits are stratified and index-disjoint. Split-order SHA-256 fingerprints are recorded in `metadata.json`. Thresholds and sigmoid calibration use only calibration data; no holdout-based tuning was performed.

## Independent holdout evaluation

Evaluation reloaded the new staging models, reconstructed splits, calculated metrics independently of the training report, and confirmed the min-cost threshold with a sorted cumulative-count implementation. Review uses `p >= threshold`; FP cost is 5 and FN cost is 500.

| Metric (min-cost policy) | RF | XGB |
|---|---:|---:|
| roc_auc | 0.9557973315 | 0.9675969406 |
| average_precision | 0.8219691343 | 0.842561434 |
| precision | 0.7159090909 | 0.5 |
| recall | 0.8513513514 | 0.8513513514 |
| f1 | 0.7777777778 | 0.63 |
| brier | 0.0004396948138 | 0.0004590778187 |
| log_loss | 0.003259377927 | 0.003242348542 |
| ece_10_equal_width | 0.0001997017108 | 0.0002267550214 |
| review_rate | 0.00205982866 | 0.002949300126 |
| total_cost | 5625 | 5815 |
| threshold | 0.00164516528 | 0.0005568324705 |

RF confusion matrix `[[TN, FP], [FN, TP]]`: `[[42623, 25], [11, 63]]`.
Uncalibrated Brier/log-loss: `0.0004549238228` / `0.006659266549`. Calibrated values improve both. Threshold bootstrap percentiles (2.5%, 50%, 97.5%; 100 calibration resamples, seed 42): `[0.00048019283043206735, 0.0016451652802712843, 0.016850577559814426]`.

XGB confusion matrix `[[TN, FP], [FN, TP]]`: `[[42585, 63], [11, 63]]`.
Uncalibrated Brier/log-loss: `0.0004885683338` / `0.003439402689`. Calibrated values improve both. Threshold bootstrap percentiles (2.5%, 50%, 97.5%; 100 calibration resamples, seed 42): `[0.0005568324704654515, 0.0005568324704654515, 0.0008002998074516654]`.

The constant training-prevalence baseline has Brier 0.0017291284 and log-loss 0.0127442145; always approving costs 37000 on this holdout. Both models outperform these simple baselines. These are reference results, not production acceptance or proof of fairness, robustness, temporal generalization or certified calibration. ECE uses ten equal-width bins and can hide rare-event miscalibration.

### Threshold changes and uncertainty

RF min-cost threshold changes from 0.0534831589433206 to 0.0016451652802712843; XGB changes from 0.22812701761722565 to 0.0005568324704654515. New calibration and cost-sensitive model fits explain different operating points; legacy training data/splits cannot be verified, so the shift cannot be attributed solely to the library update. The independent selector matches the training selector exactly. RF bootstrap variability is substantial: only 74 calibration positives are available. Thresholds are empirical choices, not stable production optima.

At the unchanged balanced presets RF precision/recall is 0.9643/0.7297 and XGB 0.9077/0.7973. Min-cost improves recall to 0.8514 while lowering precision to 0.7159/0.5000. These tradeoffs follow the declared cost objective; F1 is not the threshold-selection objective. Full per-policy and calibration comparisons are embedded in `metadata.json`. No historical performance parity is claimed.

## Regression and security evidence

Golden probe rows remain the existing synthetic contract witnesses, not original dataset records. New calibrated outputs were independently checked using the raw estimators and the fitted sigmoid coefficients before fixture update. Existing probability tolerances are preserved. API schemas, feature names/order and RF/XGB model keys remain unchanged. Probabilities, decisions at a given policy, min-cost thresholds and policy version intentionally change; callers must inspect the returned policy/threshold. Existing persisted predictions are not rewritten.

The isolated candidate passed installation, pip check, and no-exception pip-audit (142 packages; zero skipped packages or findings). The final project audit after promotion also passed without exceptions. `security-exceptions.json` is empty. The earlier numeric graph guard remains regression coverage. Runtime and integrity gates are unchanged.

RF/XGB prefit calibration and MLflow 3.17.2 native/pyfunc logging, loading and registry aliases were verified in the existing isolated compatibility experiment. It used real bundled Wisconsin data solely for library interoperability, not fraud training. MLflow uses the repository's explicit cloudpickle format; default skops serialization is not assumed compatible.

## Manifest

| File | SHA-256 |
|---|---|
| metadata.json | `277e35bc7d027bd4e24f49d0bf6c54285723f6efec10c8acc5d3c4deb57461f9` |
| policy.json | `42db759331f2cedf1d15392777d1a2db28ae3496720ffc1f789fcc371fddc7c6` |
| rf_calibrated.joblib | `afa69fe77797cf1d2b9e58ea4a9fbe5a1528ef81a72becd62707b65606dda1a1` |
| rf_pipe.joblib | `687d0920e8aab5cfe63aea74c072c535cf164634837d81d4d0745177af25d521` |
| thresholds.json | `01d27c4a7a2db78c48c0eab4648e684ea19137ae930a7d21ea9a38ce058c713e` |
| xgb_calibrated.joblib | `43dd190200f64958e6e988e758f2a2355d72762b0be79d898362f9569efd7b69` |
| xgb_pipe.joblib | `786e72c36cb6ff99825509ce1f4530210706844662c344eac69b8d57aa1f38f4` |

The metadata manifest covers the other six files; the golden fixture includes all seven hashes. No raw data, archive, experiment database or local path is included in the release.

## Remaining gates

Docker context canaries, test/API/UI image builds and inspection, Compose readiness, fresh non-root volume persistence and Redis/separate-worker integration remain NOT_RUN locally because `/var/run/docker.sock` is absent. Existing blocking GitHub Actions gates must execute on the final commit. The explicit Docker allowlist is unchanged and excludes local data/experiment stores.

Actual browser UAT remains NOT_RUN: all five pages at desktop/narrow widths, CSV errors, RF/XGB API-backed inference with fallbacks disabled, policy/manual thresholds and persisted results. Streamlit component tests are separate evidence and do not establish browser usability.

## Final local verification (2026-11-03)

- Fresh Python candidate installation and `pip check`: PASS. Final editable-project installation used no dependency changes.
- Blocking installed-environment audit with an empty exception policy: PASS, no known vulnerabilities. The prior raw no-exception audit covered 142 third-party/bootstrap packages with zero skipped packages.
- Full suite: **127 passed, 6 warnings, 80.75% coverage**, above the 75% gate. Includes MLflow tracking/registry, Streamlit components, numeric model graph/golden hashes, invalid-artifact fail-closed cases and worker/SQLite tests.
- New actual-model API/worker regression: both RF and XGB, balanced/min-cost/manual 0/manual 1, ready endpoint, persisted batch completion and 40 prediction rows: PASS.
- Ruff lint/format and `git diff --check`: PASS. JSON release files use LF bytes so Git normalization preserves their checksum manifest.
- Explicit Docker build-input inventory: PASS (81 authorized files). Context/actual image filtering, container execution, Redis integration and browser acceptance remain NOT_RUN as specified above.

The migration commit is a local review candidate. No GitHub alert resolution, main-branch remediation, release, deployment or production-quality equivalence is claimed.
