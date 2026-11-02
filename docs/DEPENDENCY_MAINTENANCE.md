# Dependency maintenance

## Scope

The local maintenance set incorporates the intended updates from Dependabot #83–#87: pytest 9.0.3, Pillow 13.0.0, FastAPI 0.162.0, Starlette 1.10.0, pre-commit 4.7.0, and MLflow 3.17.2. Streamlit 1.66.1 permits Pillow 13; 1.54.0 requires Pillow <13. Python remains 3.11 and all six serialized-model/data dependency pins remain unchanged. Packaged artifacts must not be regenerated during dependency maintenance.

Uvicorn moves from 0.30.6 to 0.52.0 because Streamlit 1.66.1 uses its Starlette server and warns that older Uvicorn with websockets 17 can abort the stream handshake when a proxy forwards non-ASCII headers. This is a compatibility dependency of the Streamlit upgrade, not an unrelated refresh.

Application and MLflow version updates share a group because their FastAPI/Starlette constraints overlap; development tools have a separate group. Security updates use their own group and are not restricted to the monthly version-update schedule. The ML-stack ignores cover version-update types only. A security PR touching that stack is a request for coordinated compatibility review, not permission to replace the runtime or artifacts automatically. CI must pass on the combined result; green individual PRs are insufficient. Closing superseded remote PRs is a separate owner action.

## Reproducibility decision

Keep the existing exact direct pins and pip workflow for this bounded change. A full transitive lock would improve repeatability, but a Windows freeze is not a portable Linux lock: platform-specific MLflow servers, native wheels, and XGBoost dependencies differ. Introducing a new resolver and maintaining multiple generated locks would expand this change and duplicate Dependabot's current manifest workflow.

CI records `pip freeze --all` in its logs, runs `pip check`, and audits the actual combined installed environment. These records are evidence, **not a lock or a promise of bit-for-bit builds**. Transitive versions and container base images can still drift. If immutable releases become a requirement, introduce generated, validated runtime/development locks for the supported platforms in a dedicated change; do not hand-edit or commit a workstation freeze as a portable lock.

## Required validation

Use a clean Python 3.11 environment:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -r requirements-dev.txt -e .
python -m pip check
python -m ruff format --check .
python -m ruff check .
python -m pytest -q --cov=src --cov-fail-under=75
python scripts/audit_dependencies.py
pre-commit validate-config
docker build --target test -t fraud-risk-ops:test .
docker compose up -d --build
```

Verify `/ready` returns 200 with compatibility fallback disabled, real predictions for both packaged models, and a worker-backed batch reaching a terminal state. Compare probabilities on fixed numeric inputs against the pre-update environment and check all manifest hashes. Verify all Streamlit pages and prediction controls. Exercise MLflow tracking, registration, aliases, rollback and loading; also open a disposable copy of a 3.16.1 store and load models logged by that version under 3.17.2. Never migrate the only copy of a real store for validation.

Security exceptions are documented in `SECURITY.md` and enforced by `scripts/audit_dependencies.py`. A failure is not resolved by setting `continue-on-error`, disabling artifact checks, or ignoring a whole package.

## Local verification, 2026-11-02

- Windows Python 3.11.9: combined installation, `pip check`, Ruff lint/format, and pre-commit configuration validation passed. Upgrade the virtualenv bootstrap tools as shown above; its original pip 24.0 had separate advisories that were resolved by upgrading pip.
- Full suite: **82 passed**, coverage **78.62%**, with the existing 75% gate. This includes real MLflow tracking/registry tests, the packaged-model baseline regression, security-policy failure cases, and Streamlit rendering/model/manual-threshold controls.
- Both packaged models returned exactly identical probabilities on 12 fixed records before and after the dependency changes. All seven packaged file hashes, including metadata, remained unchanged. The four-record test fixture was captured from main at `ca19944f4b4a578ac738005bf33ba168011d256f` before upgrading.
- A disposable MLflow 3.16.1 SQLite store and RF/XGB registered models were created before upgrading. Version 3.17.2 retained run data and champion aliases, loaded both models, and preserved exact native probabilities and default pyfunc predictions. Loading correctly warns that the logged environment specifies MLflow 3.16.1; no warning was suppressed.
- The blocking installed-environment security gate passed with only the documented scikit-learn exception. Linux Python 3.11 dependency resolution also passed; all 140 packages in that resolved graph were audited with only the same accepted finding. This resolution/audit does not substitute for a Linux installation or container test.
- Live local API: `/ready` returned 200 with strict artifact checks and compatibility fallback disabled; RF/XGB prediction parity, a worker-processed job through SQLite polling, and persisted audit records passed. Streamlit health returned 200 and AppTest rendered all five tabs in API mode. Temporary local servers were stopped after verification.
- **Blocked:** Docker test-stage build and Compose runtime verification could not run because the Ubuntu Docker daemon socket `/var/run/docker.sock` is absent; starting the daemon requires an unavailable sudo password. `docker compose config --quiet` passed. The real Redis-backed path remains unverified; only SQLite worker fallback was exercised locally.
- **Blocked:** no local browser surface is connected, so browser visual/responsive checks and file-upload interaction remain unverified. AppTest coverage is not a substitute for those checks. Existing httpx TestClient and MLflow SQLAlchemy deprecation warnings remain visible and do not fail the suite.

Owner review must retain these verification gaps until the Docker/Redis and browser checks above are executed. No remote PR, commit, push, release, or deployment was performed.
