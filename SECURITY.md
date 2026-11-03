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

## CVE-2024-5206 migration

The reference release uses `scikit-learn==1.5.2` with newly trained calibrated
RF/XGBoost models. The previous 1.4.2 artifacts are not supported by this runtime.
The [advisory](https://github.com/advisories/GHSA-jw8x-6495-233v) is fixed from 1.5.0.
`security-exceptions.json` is empty: the candidate installed graph passed a
no-exception audit before removal, and the final release audit remains blocking.
No package or advisory is suppressed by this migration.

See [migration evidence](docs/SKLEARN_MIGRATION.md) for dataset provenance,
independent holdout results, threshold changes, reproducibility and remaining
container/browser gates. Artifact runtime and checksum validation remain mandatory.
The numeric-model graph checks remain regression coverage, not an exception policy.

## Protected-mode boundary

The repository includes environment-controlled authentication for local technical review. Do not use the default credentials or API key in any exposed environment. When `REQUIRE_AUTH=true`, the application validates that local placeholder secrets were replaced. The Streamlit console can forward `X-API-Key` or bearer credentials to the API for protected local runs.

The shipped artifacts and synthetic workflows are included to make the platform runnable and inspectable. Real payment-risk decisions require organization-specific data validation, governance approval, monitoring, and review procedures around the deployed environment.
