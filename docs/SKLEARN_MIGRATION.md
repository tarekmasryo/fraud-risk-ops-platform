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
python scripts/train.py --data <verified-archive.zip> --out <new-staging-directory> --seed 42 --calibration-size 0.15 --test-size 0.15 --cost-fp 1 --cost-fn 10 --no-tracking
```

Use a new directory; never train directly into the packaged artifacts. The seven files were promoted only after independent evaluation. The model release identity is `fraud-risk-ops-v0.1.0-sklearn1.5.2`; the corrected policy identity is `fraud-risk-ops-v0.1.0-sklearn1.5.2-policy1-10`. Post-training JSON metadata adds the provenance, runtime, split fingerprints and independently computed metrics reported here. Model binaries are direct training outputs. Seeds and split hashes establish repeatability; bitwise equivalence across operating systems/BLAS implementations has not been established.

| Split | Rows | Non-fraud | Fraud |
|---|---:|---:|---:|
| train | 199364 | 199020 | 344 |
| calibration | 42721 | 42647 | 74 |
| holdout | 42722 | 42648 | 74 |

Splits are stratified and index-disjoint. Split-order SHA-256 fingerprints are recorded in `metadata.json`. Thresholds and sigmoid calibration use only calibration data; no holdout-based tuning was performed.

## Independent holdout evaluation after policy review

PR #92 review identified an unintended cost-contract change in commit `0277e2d`. This revision restores the shipped FP=1/FN=10 economics. Fitting RF/XGB and sigmoid calibrators does not depend on these costs: the four model binaries are byte-for-byte unchanged. No model training was repeated. The initial migration invocation used 5/500; that historical fact is retained in metadata, but the reproduction command above explicitly uses the restored 1/10 policy objective.

The verified original archive and exact stratified split fingerprints were checked again. Thresholds were selected only on the original 42721-row calibration split (74 positives), using sorted cumulative counts and independently cross-checked against the existing training selector. Review uses `p >= threshold`; cost is `FP + 10*FN`. The unchanged 42722-row holdout (74 positives) was evaluated only after selection.

| Metric (corrected min-cost policy) | RF | XGB |
|---|---:|---:|
| roc_auc | 0.9557973315 | 0.9675969406 |
| average_precision | 0.8219691343 | 0.842561434 |
| precision | 0.7159090909 | 0.8823529412 |
| recall | 0.8513513514 | 0.8108108108 |
| f1 | 0.7777777778 | 0.8450704225 |
| brier | 0.0004396948138 | 0.0004590778187 |
| log_loss | 0.003259377927 | 0.003242348542 |
| ece_10_equal_width | 0.0001997017108 | 0.0002267550214 |
| review_rate | 0.00205982866 | 0.001591685783 |
| total_cost | 135 | 148 |
| average_cost | 0.003159964421 | 0.003464257291 |
| threshold | 0.00164516528 | 0.3197334111 |

RF confusion matrix `[[TN, FP], [FN, TP]]`: `[[42623, 25], [11, 63]]`.
Under the same restored 1/10 costs, the reviewed commit's operating point cost `135` with precision/recall/F1 `0.715909/0.851351/0.777778` and review rate `0.00205983`. The corrected operating point costs `135`. These comparisons use the same new models and holdout, not historical models.
Calibration-only bootstrap threshold percentiles (2.5%, 50%, 97.5%; 100 resamples, seed 42): `[0.0016451652802712843, 0.003440393323615217, 0.10965628072073792]`.

XGB confusion matrix `[[TN, FP], [FN, TP]]`: `[[42640, 8], [14, 60]]`.
Under the same restored 1/10 costs, the reviewed commit's operating point cost `173` with precision/recall/F1 `0.500000/0.851351/0.630000` and review rate `0.00294930`. The corrected operating point costs `148`. These comparisons use the same new models and holdout, not historical models.
Calibration-only bootstrap threshold percentiles (2.5%, 50%, 97.5%; 100 resamples, seed 42): `[0.0005568324704654515, 0.3197334110736847, 0.6889373660087585]`.

RF retains threshold 0.0016451652802712843 even under the restored costs: it is independently confirmed as the calibration minimizer. XGB changes from 0.0005568324704654515 to 0.3197334110736847. XGB holdout cost changes from 173 to 148, with fewer false positives and lower recall. This observed outcome was not used to tune thresholds. RF cost is 135. Always approving costs 740 on this holdout.

Both bootstrap intervals are wide because calibration contains only 74 fraud cases. These are empirical reference thresholds, not stable production optima. Holdout ROC-AUC, average precision, Brier, log-loss and ECE are numerically unchanged from the reviewed commit because models and probabilities did not change. Full per-policy decision metrics and cost comparisons are embedded in metadata. No equivalence to historical models or certified operational quality is claimed.

## Policy compatibility

The public `decision_contract`, `risk_bands`, and reference-scope `metrics_boundary` are restored with the exact semantics shipped on the baseline main. `thresholds.json` retains detailed calibration/holdout metrics and restores policy-derived cost/threshold compatibility aliases. `LocalPredictor.thresholds()` now reads the validated bundle's effective merged thresholds, matching API metadata. Regression coverage verifies all four policies for both real models, sidebar defaults, costs, probability/label consistency and `GET /v1/policies` metadata. No API/worker interfaces, risk-band cutoffs or decision semantics change.

## Regression and security evidence

Golden probe rows remain the existing synthetic contract witnesses, not original dataset records. New calibrated outputs were independently checked using the raw estimators and the fitted sigmoid coefficients before fixture update. Existing probability tolerances are preserved. API schemas, feature names/order and RF/XGB model keys remain unchanged. Probabilities, decisions at a given policy, min-cost thresholds and policy version intentionally change; callers must inspect the returned policy/threshold. Existing persisted predictions are not rewritten.

The isolated candidate passed installation, pip check, and no-exception pip-audit (142 packages; zero skipped packages or findings). The final project audit after promotion also passed without exceptions. `security-exceptions.json` is empty. The earlier numeric graph guard remains regression coverage. Runtime and integrity gates are unchanged.

RF/XGB prefit calibration and MLflow 3.17.2 native/pyfunc logging, loading and registry aliases were verified in the existing isolated compatibility experiment. It used real bundled Wisconsin data solely for library interoperability, not fraud training. MLflow uses the repository's explicit cloudpickle format; default skops serialization is not assumed compatible.

## Manifest

| File | SHA-256 |
|---|---|
| metadata.json | `573a0077b7d8f5132eaa2eb2bb71213b6b39059ea019c1447ab5616bc08a02fd` |
| policy.json | `c670912b8848b905c4b3f8ce4ecd9d046247866ff29a8483713d8b555ca3d0ef` |
| rf_calibrated.joblib | `afa69fe77797cf1d2b9e58ea4a9fbe5a1528ef81a72becd62707b65606dda1a1` |
| rf_pipe.joblib | `687d0920e8aab5cfe63aea74c072c535cf164634837d81d4d0745177af25d521` |
| thresholds.json | `48dfecebb6c9e07cbe0d64b5d3e3c34d94b501edaf424ac5537e1ba23b16574e` |
| xgb_calibrated.joblib | `43dd190200f64958e6e988e758f2a2355d72762b0be79d898362f9569efd7b69` |
| xgb_pipe.joblib | `786e72c36cb6ff99825509ce1f4530210706844662c344eac69b8d57aa1f38f4` |

The metadata manifest covers the other six files; the golden fixture includes all seven hashes. No raw data, archive, experiment database or local path is included in the release.

## Remaining gates

Docker context canaries, test/API/UI image builds and inspection, Compose readiness, fresh non-root volume persistence and Redis/separate-worker integration remain NOT_RUN locally because `/var/run/docker.sock` is absent. Existing blocking GitHub Actions gates must execute on the final commit. The explicit Docker allowlist is unchanged and excludes local data/experiment stores.

Actual browser UAT remains NOT_RUN: all five pages at desktop/narrow widths, CSV errors, RF/XGB API-backed inference with fallbacks disabled, policy/manual thresholds and persisted results. Streamlit component tests are separate evidence and do not establish browser usability.

## Original migration local verification (2026-11-03; before policy correction)

- Fresh Python candidate installation and `pip check`: PASS. Final editable-project installation used no dependency changes.
- Blocking installed-environment audit with an empty exception policy: PASS, no known vulnerabilities. The prior raw no-exception audit covered 142 third-party/bootstrap packages with zero skipped packages.
- Full suite: **127 passed, 6 warnings, 80.75% coverage**, above the 75% gate. Includes MLflow tracking/registry, Streamlit components, numeric model graph/golden hashes, invalid-artifact fail-closed cases and worker/SQLite tests.
- New actual-model API/worker regression: both RF and XGB, balanced/min-cost/manual 0/manual 1, ready endpoint, persisted batch completion and 40 prediction rows: PASS.
- Ruff lint/format and `git diff --check`: PASS. JSON release files use LF bytes so Git normalization preserves their checksum manifest.
- Explicit Docker build-input inventory: PASS (81 authorized files). Context/actual image filtering, container execution, Redis integration and browser acceptance remain NOT_RUN as specified above.

The migration commit is a local review candidate. No GitHub alert resolution, main-branch remediation, release, deployment or production-quality equivalence is claimed.

## Policy correction local verification

- Full suite after the PR #92 correction: **128 passed, 6 warnings, 80.97% coverage**, above the unchanged 75% gate.
- Ruff lint/format, `git diff --check`, real-model API/worker persistence, artifact fail-closed/integrity tests and no-exception security audit: PASS.
- New Local/API regression verifies effective costs, all four RF/XGB policy thresholds, legacy UI aliases/sidebar defaults, probabilities/labels and the exact public decision-contract, risk-band and reference-boundary semantics.
- Four model binaries, golden input rows and golden probability values match reviewed commit `0277e2d` exactly. Only policy-related JSON/hash witnesses and the local threshold accessor change behavior.
- Explicit Docker allowlist is unchanged. Actual Docker/image/Compose/Redis and browser acceptance remain NOT_RUN locally; final-commit CI and manual browser gates still apply. No push or workflow dispatch is included in this local correction.
