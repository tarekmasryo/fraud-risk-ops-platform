# Reference application hardening

This change is separate from dependency maintenance. It does not change model
files, dependencies, architecture, or worker crash recovery.

## Runtime and artifact contract

API and worker scoring validate runtime versions and SHA-256 manifest entries
before deserializing models. Missing metadata, missing manifest entries, missing
files, checksum mismatches, and incompatible runtimes fail closed. Compatibility
fallback settings cannot bypass this load-time validation. The API returns 503;
batch jobs record failure without predictions.

Validated scoring bundles, including policies, are retained per process. Deploy
an immutable, trusted artifact directory and restart API and worker processes
when replacing it. Hashing is not repeated per prediction request. Readiness
still checks files on disk. A cached, validated model is not replaced by a changed
file until restart. The manifest detects corruption; it is not a signature and
does not protect against an attacker replacing both metadata and model files.

## Local containers

The Docker context defaults to deny and explicitly includes code, test fixtures,
configuration needed by the image, and the seven packaged release artifacts.
Local datasets, state, credentials, temporary files, and unrelated directories
are excluded. Do not embed credentials in source files. New build inputs must be
explicitly added to `.dockerignore` after review. The API image creates writable
`/app/data` for a fresh named volume; existing volumes are preserved.

Each ignore exception names one file. Directory exceptions and wildcard
exceptions are rejected by the context gate. Adding a Python source/test/script
file without authorizing it also fails the input inventory test.

`python scripts/check_docker_context.py` copies only authorized inputs into a
disposable directory and adds fictional canaries, including JSON configuration,
an extra model, unapproved Python files, CSVs, SQLite state, `.env` and temporary
files. Docker builds a scratch image with `COPY . /context/` and exports its
filesystem. The gate requires exactly the authorized files, with matching hashes;
any leaked canary, other unexpected file, missing input or changed content fails.
No Python glob matcher substitutes for Docker filtering. The disposable directory
is removed on completion or failure; missing Docker is a failure, never a pass.

The Docker CI job runs this gate before the existing test-stage build and Compose
smoke test. It then inspects the test, API, worker and UI images with
`--images`, requiring all authorized inputs with their original hashes and the
absence of canary paths. The Redis/separate-worker check remains blocking.

Published demo ports bind to `127.0.0.1`. Container-to-container service names
and ports remain unchanged. Remote access now requires an intentional override
and appropriate authentication; the default Compose stack is a local demo.

## UI and time semantics

Malformed, empty, nonnumeric, non-finite, invalid-label, and missing-feature CSVs
produce actionable messages. Failed prediction input clears the previous run;
failed profiling uploads do not silently use another dataset. Profiling uses
prediction metrics only when dataset values, columns, index and row order match
the prediction dataset. Conservative mismatches require running prediction again.

Single-class datasets can still be scored. ROC/AUC threshold analysis and
ROC-AUC permutation importance require both classes and otherwise display an
explanation. The sampled importance dataset is checked as well.

`Time` remains elapsed seconds since the first transaction. The elapsed-hour
modulo-24 distribution is retained with an explicit explanation. Business-hour
and night proxies are removed. The API response schema, probabilities, labels,
and policy decisions are unchanged, but `unusual_time_window` is no longer emitted
in `reason_codes`. Consumers must not require that optional hint. Previously
persisted records are not rewritten.

## Verification gates

- Run Ruff and the complete Pytest suite with `--cov=src --cov-fail-under=75`.
  Packaged-artifact golden tests verify hashes and RF/XGBoost probability parity.
- On a Docker host or GitHub Actions, build the test and runtime images, run the
  existing Compose readiness/metrics and Redis separate-worker integration checks
  with a fresh disposable volume. Confirm the non-root API can persist records.
- Inspect the built context/image with disposable sentinels: root `creditcard.csv`,
  `data/private.csv`, `.tmp/private.txt`, `.env`, a nested `.env`, and a local SQLite
  database must be absent. Source, regression fixtures and packaged models must
  remain present. Use a disposable checkout and never real secrets for this check.
- Actual browser UAT remains a separate gate: API-backed inference with both
  fallbacks disabled; all five tabs at desktop and narrow widths; valid, malformed,
  empty, missing-feature and nonnumeric CSVs; RF/XGBoost and policy/manual threshold
  controls; profiling with matching, reordered, equal-size different and shorter
  datasets; all-zero/all-one labels; browser console and application logs. Component
  tests do not establish browser rendering, upload interaction or usability.

Worker crash recovery remains tracked in [WORKER_RECOVERY.md](WORKER_RECOVERY.md).

## Local validation before context correction (2026-11-02)

On Python 3.11.9 with the existing maintenance environment:

- Ruff lint and format checks passed; `git diff --check` passed.
- Full suite: 117 passed, 6 warnings; coverage 80.47% (required 75%). The existing
  coverage configuration excludes UI code; separate UI/component tests passed.
- Golden checks confirmed unchanged packaged hashes and RF/XGBoost predictions.
- Tests confirmed API 503 and failed worker jobs for invalid artifacts before
  deserialization, plus cached-bundle reuse, CSV errors, dataset matching and
  single-class evaluation guards.
- Docker integration was not run: the WSL Docker client could not connect because
  `/var/run/docker.sock` is absent. Compose/build and browser gates above remain
  required; no GitHub Actions run was initiated for these uncommitted changes.

The six warnings are existing Starlette TestClient and MLflow/SQLAlchemy warnings.

## Targeted context correction validation (2026-11-02)

- 44 focused tests passed (`test_docker_context.py` and
  `test_reference_hardening.py`). These test the inventory and gate failure
  detection as well as existing hardening regressions; they do not establish
  Docker filtering behavior locally.
- Ruff lint/format and `git diff --check` passed. The allowlist authorizes 81 exact
  files. CI wiring includes canaries, image checks and the existing Redis gate.
- Compose configuration validation passed through the WSL Docker Compose client.
- Actual context export, image builds/inspection and Compose/Redis execution
  remain unrun: the Docker daemon socket is absent. GitHub Actions must run these
  blocking gates before merge. No workflow was triggered by this local change.

Final local pre-commit validation after the context correction: the complete
suite passed with 126 tests, 6 existing warnings and 80.47% coverage against the
75% gate. Ruff lint/format and `git diff --check` passed. All seven packaged
artifact hashes remain unchanged. Docker execution and actual browser acceptance
remain NOT_RUN; the CI and browser gates above still apply.
