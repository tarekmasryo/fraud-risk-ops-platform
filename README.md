# 🛡️ Fraud Risk Ops Platform

[![CI](https://github.com/tarekmasryo/fraud-risk-ops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/tarekmasryo/fraud-risk-ops-platform/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-risk%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-review%20console-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](./VERSION)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Production-structured fraud risk operations platform** for fraud scoring, policy-driven decisions, review workflows, audit logging, worker-backed batch jobs, and monitoring-ready signals.

The system models the operational boundary around an ML risk decision:

```text
transaction record -> schema validation -> model score -> policy decision -> audit log -> review console -> metrics
```

---

## 🎯 Why this project matters

Fraud scoring is not only a model problem. A useful operational system needs stable contracts, policy review, auditability, batch processing, and visibility into runtime behavior.

This project demonstrates those production-minded boundaries around an ML decision system:

- explicit API contracts
- model/policy separation
- strict input validation
- fail-closed artifact readiness
- persisted audit trails
- worker-backed batch-job lifecycle
- Prometheus/Grafana observability hooks
- documented engineering trade-offs

---

## ✨ What this repo demonstrates

| Layer | Implementation |
|---|---|
| **Risk API** | FastAPI `/v1` endpoints for prediction, policy, jobs, audit logs, and model metadata. |
| **Review Console** | Streamlit UI for scoring, threshold review, model views, data quality checks, and operational slices. API mode chunks large batches automatically to respect backend limits. |
| **Policy Governance** | `artifacts/policy.json` is the threshold source of truth for API and UI; UI policy/manual threshold selections are forwarded to the API in API mode. |
| **Decision Contract** | Responses include `risk_score`, `decision`, `review_required`, `risk_band`, `reason_codes`, `policy_version`, and `input_hash`. |
| **Readiness Gate** | `/ready` validates metadata, schema, policy, model files, runtime compatibility, checksums, and sample scoring. |
| **Persistence** | SQLite operations store for prediction requests, row-level predictions, audit logs, and batch jobs. |
| **Batch Work** | Docker Compose runs API + worker; SQLite is source of truth and Redis is a wake-up queue. |
| **Observability** | `/metrics`, `/v1/metrics/summary`, Prometheus config, and provisioned Grafana dashboard. |
| **Quality Gates** | Ruff, Pytest, coverage config, Dockerfile, Docker Compose, `.dockerignore`, `.gitignore`, and CI workflow. |

---

## 🧭 Architecture

```mermaid
flowchart LR
  Client["API Client"] --> API["FastAPI Risk API"]
  UI["Streamlit Review Console"] --> API
  API --> Auth["Auth Guard"]
  API --> Validator["Schema Validator"]
  Validator --> Engine["Risk Decision Engine"]
  Engine --> Models["Model Artifacts"]
  Engine --> Policy["Policy Service"]
  Policy --> PolicyFile["artifacts/policy.json"]
  Engine --> Store[("SQLite Ops Store")]
  Store --> Audit["Audit Trail"]
  API --> Queue["Redis Wake-up Queue"]
  Queue --> Worker["Batch Worker"]
  Worker --> Store
  API --> Metrics["Prometheus Metrics"]
  Metrics --> Grafana["Grafana Dashboard"]
```

Core package map:

```text
src/fraud_dashboard/
├─ api/                 # FastAPI app and versioned API surface
├─ core/                # config, policy, decision logic, artifacts, security, readiness
├─ data/                # synthetic data and validation helpers
├─ observability/       # Prometheus metrics
├─ platform/            # SQLite store, jobs, audit persistence, Redis wake-up queue
├─ services/            # application services for scoring, jobs, reference data, and summaries
├─ ui/                  # Streamlit review console
└─ workers/             # batch worker entrypoint
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`docs/ENGINEERING.md`](docs/ENGINEERING.md).

---

## 🖼️ Screenshots

| Data Overview | Prediction Engine | Model Metrics |
|---|---|---|
| ![](assets/data_overview.png) | ![](assets/prediction_engine.png) | ![](assets/model_metrics.png) |

| Model Insights | Data Quality |
|---|---|
| ![](assets/model_insights.png) | ![](assets/data_quality.png) |

---

## 🚀 Quickstart with Docker Compose

```bash
docker compose up --build
```

Open:

| Service | URL |
|---|---|
| API docs | `http://127.0.0.1:8000/docs` |
| Review console | `http://127.0.0.1:8501` |
| Prometheus | `http://127.0.0.1:9090` |
| Grafana | `http://127.0.0.1:3000` |

Stop:

```bash
docker compose down
```

Remove local Docker volumes when you want a clean runtime store:

```bash
docker compose down -v
```

If Docker Compose fails with `Bind for 0.0.0.0:6379 failed: port is already allocated`, another Redis instance is already using the host port. Redis is only required inside the Docker network by the API and worker, so you can remove the Redis host `ports` mapping or change it to `6380:6379`.

---

## 🧪 Quickstart locally

> The official runtime target is Python 3.11. Patch-level Python 3.11 differences may appear as readiness warnings when artifact checks and sample scoring pass.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Run API:

```bash
python api.py
```

Run UI:

```bash
FRAUD_API_URL=http://127.0.0.1:8000 python -m streamlit run app.py --server.port 8501
```

Any available Streamlit port is fine. For example, use `--server.port 8503` if `8501` is already in use. The important part is that `FRAUD_API_URL` points to the running API.

Run quality checks:

```bash
ruff format --check .
ruff check .
pytest -q --cov=src --cov-fail-under=75
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt -r requirements-dev.txt
$env:PYTHONPATH="src"
$env:REQUIRE_AUTH="false"
python -m uvicorn fraud_dashboard.api.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="src"
$env:FRAUD_API_URL="http://127.0.0.1:8000"
python -m streamlit run app.py --server.port 8501
```

---

## 🔌 API surface

Runtime checks:

```text
GET /live
GET /ready
GET /metadata
GET /metrics
```

Versioned platform API:

```text
POST /v1/auth/login
GET  /v1/me
POST /v1/predictions
POST /v1/predictions/batch
POST /v1/batch-jobs
GET  /v1/jobs/{job_id}
GET  /v1/audit-logs
GET  /v1/policies
GET  /v1/metrics/summary
GET  /v1/model-versions
```

Prediction requests use the packaged credit-card fraud feature contract: `Time`, `V1` through `V28`, and `Amount`. The default model key is `rf`; `xgb` is also available when the matching artifact is present. Use `POST /v1/predictions` for one record and `POST /v1/predictions/batch` for multiple records.

For interactive request testing, open `http://127.0.0.1:8000/docs`. For the full request and response contract, see [`docs/API.md`](docs/API.md).

Example response shape:

```json
{
  "model": "rf",
  "threshold": 0.0534831589433206,
  "proba_fraud": 0.0123,
  "risk_score": 0.0123,
  "label": 0,
  "decision": "approve",
  "review_required": false,
  "risk_band": "low",
  "reason_codes": ["score_below_policy_threshold", "risk_band_low"],
  "policy": "min_cost",
  "policy_version": "fraud-risk-ops-v0.1.0",
  "input_hash": "...",
  "latency_ms": 12,
  "request_id": "pred_..."
}
```

---

## 🧩 Policy governance

`artifacts/policy.json` is the source of truth for operating thresholds. The API and UI read the same policy file to avoid backend/UI drift. In API mode, the Streamlit console forwards the selected policy preset or manual threshold to the API, then chunks large scoring runs according to the backend `MAX_BATCH_RECORDS` limit.

Current operating policies:

| Policy | Intent |
|---|---|
| `strict` | Reduce false positives and analyst load. |
| `balanced` | Default review-oriented operating point. |
| `min_cost` | Cost-aware packaged reference operating point; regenerate artifacts for target-data holdout metrics. |
| `lenient` | Increase fraud capture with higher review load. |

Explicit invalid policy requests fail with `400` instead of silently falling back to another threshold. That behavior is deliberate: policy selection is part of the decision contract.

---

## ✅ Artifact readiness

Use:

```text
GET /ready
```

The readiness check validates:

- metadata presence
- schema feature contract
- policy file presence
- model artifact presence
- artifact checksums against the expected SHA-256 manifest in `metadata.json`
- runtime compatibility against metadata
- sample scoring without compatibility fallback

Readiness fails closed for missing artifacts, checksum mismatches, schema/policy errors, or incompatible runtime failures. Patch-level Python 3.11 differences may be reported as warnings when artifact checks and sample scoring pass. The API does not silently replace a broken serialized model with heuristic scores.

---

## ⚙️ Configuration

Copy `.env.example` and adjust values as needed.

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | Local operations persistence store | `sqlite:///./data/fraud_ops.db` |
| `MAX_BATCH_RECORDS` | Batch scoring safety limit | `1000` |
| `RUN_JOBS_IN_API` | Run jobs inside API process for one-process local runs | `true` |
| `WORKER_POLL_SECONDS` | Worker poll interval when Redis wake-up is unavailable | `2` |
| `APP_ENV` | Runtime environment guard; `prod` requires auth | `dev` |
| `REQUIRE_AUTH` | Protect scoring and operational endpoints | `false` |
| `CORS_ALLOW_ORIGINS` | Comma-separated browser origins allowed for API clients | local Streamlit origins |
| `DEMO_API_KEY` / `DEMO_API_KEY_HASH` | Backend API key or HMAC digest when auth is enabled | replace before protected runs |
| `FRAUD_API_KEY` / `FRAUD_BEARER_TOKEN` | Optional Streamlit-to-API auth forwarding | empty |
| `STRICT_ARTIFACT_RUNTIME` | Fail `/ready` on incompatible artifact/runtime failures | `true` |
| `ALLOW_ARTIFACT_COMPATIBILITY_FALLBACK` | Opt-in local UI compatibility fallback only | `false` |
| `ALLOW_LOCAL_FALLBACK` | Allow UI to use local artifacts if API is down | `true` |
| `PROMETHEUS_ENABLED` | Expose Prometheus metrics endpoint | `true` |

For protected local runs, set strong non-default values:

```bash
REQUIRE_AUTH=true
JWT_SECRET_KEY=<strong-secret>
API_KEY_HASH_SECRET=<strong-hmac-secret>
DEMO_API_KEY=<strong-api-key>
# optional: DEMO_API_KEY_HASH=<hmac-sha256-api-key-digest>
ADMIN_PASSWORD=<strong-password>
# optional for Streamlit-to-API protected mode:
FRAUD_API_KEY=<same-strong-api-key>
GRAFANA_ADMIN_PASSWORD=<strong-local-grafana-password>
```

The app refuses insecure local secrets when `REQUIRE_AUTH=true`. It also refuses `APP_ENV=prod` unless auth is enabled, and rejects wildcard CORS in prod-like environments. Scoring, operational endpoints, and `/metadata` require authentication in protected mode. The Streamlit console can also forward protected-mode credentials through `FRAUD_API_KEY`, `FRAUD_BEARER_TOKEN`, or the sidebar auth fields.

---

## 🧠 Training and metric boundary

The packaged artifacts are reference artifacts for running and reviewing the platform. They are not presented as operational benchmark claims. To regenerate deployable model artifacts, run `scripts/train.py`; the script uses separate splits for model training, calibration/threshold selection, and final holdout-test metric reporting.

```bash
python scripts/train.py --data data/creditcard.csv --out artifacts --label Class
```

The resulting `metadata.json` records the split contract and stores expected SHA-256 checksums for readiness validation.

### Local experiment tracking

Install both `requirements.txt` and `requirements-dev.txt` as shown above. Training tracks
one MLflow run in the `fraud-risk-training` experiment by default; no tracking server is
required. Run these commands from the repository root:

```bash
python scripts/train.py --data data/creditcard.csv --out artifacts --label Class
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Run metadata lives in `mlflow.db` (SQLite), and logged
artifact copies live in `mlruns/`, relative to the working directory. Both are ignored
by Git and Docker. Existing files in `--out` retain their names, schemas, and serialization.

Each run records the seed, split fractions/counts, costs, label, feature count, dataset
basename and SHA-256 (no raw dataset or full dataset path), and estimator parameters under
`rf.*` and `xgb.*`. Metrics use `rf.calibration.*`, `rf.holdout_test.*`,
`xgb.calibration.*`, and `xgb.holdout_test.*`: threshold, precision, recall, F1,
ROC AUC when defined, review rate, total/average cost, and TP/FP/TN/FN. Calibration
metrics describe threshold selection; only holdout metrics report final performance.
After all release files are written, the run logs `thresholds.json`, `policy.json`,
`metadata.json`, and the four `.joblib` files under `release/`.

Use `--no-tracking` to train without MLflow, `--experiment-name NAME` to group runs
separately, or `--tracking-uri URI` to override the store. The URI flag takes precedence
over `MLFLOW_TRACKING_URI`, which takes precedence over `sqlite:///mlflow.db`.
For example, in PowerShell:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///another-experiment.db"
python scripts/train.py --data data/creditcard.csv --out artifacts
mlflow ui --backend-store-uri $env:MLFLOW_TRACKING_URI --host 127.0.0.1 --port 5000
```

A future server can be selected with the same URI setting; this setup provisions no
remote infrastructure. Tracking errors propagate, so a run with incomplete logging is
not reported as successful. Files already written to `--out` remain available if logging
fails. Keep the dataset and code revision alongside the recorded configuration to reproduce
a run; tracking does not snapshot the dataset or source tree.

### Local model registry and explicit promotion

Tracked runs also log both calibrated estimators as MLflow `sklearn` models with input
signatures and a `python_function` probability interface. No input examples or dataset
rows are logged. Registration is **opt-in**; normal training creates no registered
versions. Use the same local SQLite store, from the repository root:

```bash
python scripts/train.py --data data/creditcard.csv --out artifacts --register-models
python scripts/model_registry.py --name fraud-risk-rf list
python scripts/model_registry.py --name fraud-risk-rf aliases
python scripts/model_registry.py --name fraud-risk-rf set-candidate --version 1
python scripts/model_registry.py --name fraud-risk-rf promote
python scripts/model_registry.py --name fraud-risk-rf champion
python scripts/model_registry.py --name fraud-risk-rf smoke --input-csv data/features.csv
```

Each registration creates a new version under `fraud-risk-rf` and `fraud-risk-xgb`,
tagged with model family, run ID, dataset SHA-256, holdout F1, ROC AUC when defined,
average cost, and the calibration-selected threshold. Review these tags with `list`
before choosing a candidate. Training assigns no aliases. Only `promote` assigns
or reassigns `champion` to the current `candidate`; it never chooses a metric winner.
Repeat `set-candidate` and `promote` with a previous version to roll back.

Aliases are mutable references to immutable model versions. Moving `candidate` alone
does not move `champion`, and moving `champion` affects subsequent loads, not models
already loaded in memory. Both scripts use `--tracking-uri`, then `MLFLOW_TRACKING_URI`,
then `sqlite:///mlflow.db`; registry operations use that same store, ignoring any
separate `MLFLOW_REGISTRY_URI`. `--register-models` cannot be combined with `--no-tracking`.

Load by alias directly, or use `smoke` with a compatible feature-only CSV (same feature
names and numeric types as training, without the label):

```python
import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")
model = mlflow.pyfunc.load_model("models:/fraud-risk-rf@champion")
probabilities = model.predict(pd.read_csv("data/features.csv"))
```

The output contains calibrated probabilities in estimator class order (`[0, 1]`
for the binary fraud models). It does not apply the tagged threshold or runtime
decision policy. Filesystem artifacts and API/UI model loading remain unchanged;
promotion does not deploy or replace those artifacts. Models use `cloudpickle` to
preserve both calibrated wrappers, so load only trusted local model artifacts.
Logging/registration failures propagate and may leave an earlier version from the
same run registered; inspect the run status before promotion. This CLI is intended
for a local operator, not concurrent promotion automation.

---

## 📦 Data

This repository does not redistribute the original credit-card fraud dataset. The UI can run with synthetic data and accepts compatible uploaded CSV files.

Auto-load order:

1. `data/creditcard.csv`
2. `creditcard.csv`
3. `data/demo_creditcard.csv`
4. `/mnt/data/creditcard.csv`

See [`DATA_LICENSE.md`](DATA_LICENSE.md).

---

## 📚 Documentation

| Doc | Purpose |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System boundaries, runtime flows, governance, and deployment topology. |
| [`docs/ENGINEERING.md`](docs/ENGINEERING.md) | Clean-code boundaries, design patterns, and why the repo stays compact. |
| [`docs/API.md`](docs/API.md) | Endpoint reference and request/response contract. |
| [`docs/OPERATIONS.md`](docs/OPERATIONS.md) | Local operations, runtime checks, worker mode, and monitoring. |
| [`docs/JOB_LIFECYCLE.md`](docs/JOB_LIFECYCLE.md) | Persisted async batch-job lifecycle. |
| [`docs/DATABASE_SCHEMA.md`](docs/DATABASE_SCHEMA.md) | SQLite operational tables. |
| [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) | Engineering trade-offs behind persistence, queueing, auth, artifacts, and Docker. |
| [`docs/SCOPE.md`](docs/SCOPE.md) | Release scope and operational boundaries. |
| [`docs/CASE_STUDY.md`](docs/CASE_STUDY.md) | Short system-design framing. |
| [`SECURITY.md`](SECURITY.md) | Security policy and protected-mode boundary. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes. |

---

## 📌 Release scope

This release focuses on the engineering system around fraud-risk decisions:

- model serving through stable API contracts
- threshold policy governance
- auditable prediction and batch-job records
- seeded model-version and threshold-policy reference tables
- review-oriented Streamlit interface
- worker-backed batch processing
- Prometheus/Grafana observability
- Docker Compose deployment for local technical review

The repository uses synthetic/local runtime defaults and does not ship real customer data, payment-network integrations, tenant isolation, billing, or regulatory-compliance workflows. Those boundaries keep the release focused, runnable, and technically inspectable.

---

## 📄 License and attribution

- Code license: MIT — see [`LICENSE`](LICENSE).
- Dataset files are not redistributed. Follow original dataset terms when using external data.
