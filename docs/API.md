# API Reference

Fraud Risk Ops exposes a small versioned API for scoring, policy review, audit access, and batch-job orchestration.

The primary API surface is `/v1/*`. Legacy `/predict` routes remain for compatibility with earlier clients and use the same auth guard when `REQUIRE_AUTH=true`.

---

## Runtime checks

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/live` | Process liveness check. |
| `GET` | `/health` | Backward-compatible lightweight health check. |
| `GET` | `/ready` | Artifact, policy, runtime, checksum, and sample-scoring readiness check. |
| `GET` | `/metadata` | Runtime metadata, schema, thresholds, policy, API limits, and checksums; protected when `REQUIRE_AUTH=true`. |
| `GET` | `/metrics` | Prometheus text-format metrics. |

---

## Versioned platform API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/auth/login` | JWT login. |
| `GET` | `/v1/me` | Current principal. |
| `POST` | `/v1/predictions` | Score one record and persist an audit trail. |
| `POST` | `/v1/predictions/batch` | Score a bounded batch synchronously. `MAX_BATCH_RECORDS` is exposed through `/metadata.limits` for UI/client chunking. |
| `POST` | `/v1/batch-jobs` | Create a persisted asynchronous batch job. |
| `GET` | `/v1/jobs/{job_id}` | Read batch-job status/result metadata. |
| `GET` | `/v1/audit-logs?limit=50&offset=0` | Read audit events with simple offset pagination. |
| `GET` | `/v1/policies` | Read the active policy document. |
| `GET` | `/v1/metrics/summary` | Read persisted operational counters for dashboards and reviewers. |
| `GET` | `/v1/model-versions` | Read model metadata, seeded model-version rows, and artifact checksums. |

When `REQUIRE_AUTH=true`, scoring and operational endpoints require either a valid bearer token or `X-API-Key`.

---

## Legacy compatibility

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Legacy single-record scoring route, protected in auth-enabled mode. |
| `POST` | `/predict/batch` | Legacy synchronous batch scoring route, protected in auth-enabled mode. |

---

## Single prediction request

```json
{
  "model": "rf",
  "policy": "min_cost",
  "record": {
    "Time": 0,
    "Amount": 42.0,
    "V1": 0.0
  }
}
```

The record must include all features listed in `artifacts/metadata.json`. The example above is abbreviated for readability.

### Policy resolution rules

1. If `threshold` is supplied, it is treated as a manual override.
2. If `policy` is supplied, it must exist in `artifacts/policy.json`.
3. If neither is supplied, the default policy from `policy.json` is used.
4. Invalid explicit policy names fail with `400` and do not silently fall back.

---

## Single prediction response

```json
{
  "model": "rf",
  "threshold": 0.0016451652802712843,
  "proba_fraud": 0.000123,
  "risk_score": 0.000123,
  "label": 0,
  "decision": "approve",
  "review_required": false,
  "risk_band": "low",
  "reason_codes": [
    "score_below_policy_threshold",
    "risk_band_low"
  ],
  "policy": "min_cost",
  "policy_version": "fraud-risk-ops-v0.1.0-sklearn1.5.2-policy1-10",
  "input_hash": "...",
  "latency_ms": 12,
  "request_id": "pred_..."
}
```

`reason_codes` are deterministic operational review hints. They are not SHAP values or formal model explanations.

---

## Batch-job response

```json
{
  "job_id": "job_...",
  "status": "queued"
}
```

Read job status with:

```text
GET /v1/jobs/{job_id}
```

The persisted job response includes status, record counts, result metadata, and error information when a job fails.

---

## Metrics summary

`GET /v1/metrics/summary` returns persisted operational state from the local store. Unlike Prometheus process counters, this endpoint is backed by SQLite and can summarize completed API/worker activity from the shared runtime volume.

Example shape:

```json
{
  "prediction_requests": 12,
  "predictions": 1200,
  "high_risk_predictions": 84,
  "high_risk_rate": 0.07,
  "average_latency_ms": 18.5,
  "jobs": {
    "queued": 0,
    "running": 0,
    "completed": 3,
    "failed": 0,
    "processed_records": 1000,
    "failed_records": 0
  },
  "policy_usage": {"min_cost": 12},
  "model_usage": {"rf": 12},
  "latest_policy_version": "fraud-risk-ops-v0.1.0",
  "active_model_versions": 2
}
```

## Error behavior

| Scenario | Status | Behavior |
|---|---:|---|
| Missing required feature | `400` | Request is rejected before scoring. |
| Non-numeric feature value | `400` | Request is rejected before scoring. |
| Unknown model | `400` | Request is rejected. |
| Unknown named policy | `400` | Request is rejected. |
| Artifact/runtime scoring failure | `503` | API fails closed rather than returning heuristic scores. |
| Oversized batch | `422` | Pydantic validation rejects the request; official clients should chunk by `/metadata.limits.max_batch_records`. |

---

## Auth notes

`REQUIRE_AUTH=false` keeps local technical review frictionless. When `REQUIRE_AUTH=true`, replace local placeholder secrets. The application refuses unsafe defaults in protected mode, and `APP_ENV=prod` is rejected unless auth is enabled. Browser clients are limited by `CORS_ALLOW_ORIGINS`; wildcard CORS is rejected in prod-like environments. Scoring routes, operational routes, and `/metadata` require authentication in protected mode. API clients may authenticate with `X-API-Key`; the Streamlit console can forward the same key through `FRAUD_API_KEY` or the sidebar auth field.
