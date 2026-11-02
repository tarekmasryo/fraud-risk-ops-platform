# Operations

## Run locally with Docker Compose

```bash
docker compose up --build
```

Open:

- API docs: `http://127.0.0.1:8000/docs`
- UI: `http://127.0.0.1:8501`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`

## Runtime checks

Use `/live` for process liveness and `/ready` for artifact readiness.

`/ready` checks artifact metadata, policy presence, model files, expected SHA-256 checksums, runtime warnings, and a sample prediction with compatibility fallback disabled. A failed `/ready` response is expected when the local runtime does not match the pinned artifact environment.

## Persistence

The release uses SQLite by default for a low-friction local setup. The schema includes prediction history, audit logs, model versions, threshold policies, and batch jobs.

Docker Compose stores runtime data in the `fraud_data` volume.

## Worker mode

Local one-process runs use:

```text
RUN_JOBS_IN_API=true
```

Docker Compose sets:

```text
RUN_JOBS_IN_API=false
```

In compose mode, the API persists queued jobs, pushes a Redis wake-up event, and the worker claims and executes jobs from the SQLite-backed job table. If Redis is temporarily unavailable, the worker still polls SQLite for queued jobs.

Worker crash recovery is a separate limitation; see [WORKER_RECOVERY.md](WORKER_RECOVERY.md). Dependency update policy and validation are documented in [DEPENDENCY_MAINTENANCE.md](DEPENDENCY_MAINTENANCE.md).

## Monitoring

Prometheus scrapes the API `/metrics` endpoint. `GET /v1/metrics/summary` provides persisted operational counters for reviewers and dashboard consumers. Grafana provisions the Prometheus datasource and Fraud Risk Ops dashboard automatically from `monitoring/grafana/`; the dashboard includes HTTP latency, prediction latency, errors, high-risk rate, policy usage, and batch-job status panels.


## Artifact runtime policy

The release fails closed on model artifact/runtime mismatch by default. Keep `STRICT_ARTIFACT_RUNTIME=true` for API and Docker runs. Use `ALLOW_ARTIFACT_COMPATIBILITY_FALLBACK=true` only for local UI exploration when you knowingly accept deterministic heuristic fallback behavior.

## Protected-mode policy

When `REQUIRE_AUTH=true`, replace `JWT_SECRET_KEY`, `DEMO_API_KEY` or `DEMO_API_KEY_HASH`, and `ADMIN_PASSWORD` with strong non-default values. The application refuses unsafe placeholder defaults in protected mode. `APP_ENV=prod` also fails fast unless `REQUIRE_AUTH=true`, and wildcard CORS is rejected in prod-like environments. Scoring routes, including legacy compatibility routes, require authentication in protected mode. For the Streamlit console, set `FRAUD_API_KEY`/`FRAUD_BEARER_TOKEN` or paste the credential into the sidebar auth fields.

## Runtime cache policy

Model artifacts and `policy.json` are loaded through process-local caches. This keeps request latency stable and avoids filesystem reads on every prediction. Changing artifacts or policy files requires an API/worker restart. That behavior is intentional for this release; hot policy reload is a production extension, not an implicit side effect.

## Browser client policy

Set `CORS_ALLOW_ORIGINS` to a comma-separated allowlist when serving a browser frontend outside the local Streamlit defaults. Keep it restricted for protected or prod-like deployments.
