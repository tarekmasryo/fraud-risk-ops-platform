# Security Policy

## Reporting a vulnerability

If you believe you found a security issue, please open a private report via GitHub Security Advisories (preferred) or open an issue with **no sensitive details**.

## Dependency updates

This repository ships **pre-trained serialized artifacts** in `artifacts/`. To keep the repo runnable out of the box, the ML/data stack is pinned on `main`.

- CI runs `python scripts/audit_dependencies.py` as a **blocking** gate over the installed runtime and development environment, including transitive dependencies. Audit errors, unaudited packages, new findings, and expired exceptions fail the gate.
- Only this repository's own editable distribution is excluded from advisory lookup; third-party editable distributions remain blocking. The project's code is covered by source review and tests.
- Dependabot groups application, development, and GitHub Actions version updates. Security updates have a separate Python group. Routine version updates to `numpy`, `scipy`, `scikit-learn`, `xgboost`, `joblib`, and `pandas` are ignored; security updates remain enabled and must undergo artifact compatibility review. Repository security-update enablement is managed on GitHub, not by this file.
- `Streamlit` and `Pillow` belong to the application group, not the serialized model stack. Update them together when their constraints require it, and verify the rendered UI and existing model predictions.

When the ML/data stack changes, refresh the artifacts and rerun the full validation suite before release.

## Temporary vulnerability exception

`security-exceptions.json` accepts only **CVE-2024-5206** (aliases **PYSEC-2024-110**, **GHSA-jw8x-6495-233v**) in **scikit-learn 1.4.2**, until **2026-12-02**. The repository maintainer owns review and removal. The audit checks the package, exact version, advisory or alias, and expiration; it does not suppress other scikit-learn findings. Expiration fails CI even if the advisory disappears.

The [advisory](https://github.com/advisories/GHSA-jw8x-6495-233v) concerns training tokens retained by `TfidfVectorizer.stop_words_`. `scripts/train.py` trains numeric RandomForest/XGBoost models with calibrated wrappers and uses no text vectorizers. The packaged artifacts are trusted reference models, not user-uploaded estimators. Maintenance verification inspects their fitted object graphs for text vectorizers and compares predictions before and after the upgrade. Uploaded application input is numeric CSV, not serialized models.

This is a scoped applicability exception, not a claim that scikit-learn is generally vulnerability-free. Version 1.5.0 is the advisory's fix, but changing scikit-learn alone invalidates the pinned artifact environment. Remove the exception during a coordinated ML-stack/artifact refresh, or earlier if text vectorizers, untrusted models, or relevant data paths are introduced. Do not extend it automatically.

## Protected-mode boundary

The repository includes environment-controlled authentication for local technical review. Do not use the default credentials or API key in any exposed environment. When `REQUIRE_AUTH=true`, the application validates that local placeholder secrets were replaced. The Streamlit console can forward `X-API-Key` or bearer credentials to the API for protected local runs.

The shipped artifacts and synthetic workflows are included to make the platform runnable and inspectable. Real payment-risk decisions require organization-specific data validation, governance approval, monitoring, and review procedures around the deployed environment.
