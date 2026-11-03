# Changelog

## Unreleased — scikit-learn security and reference model migration

- Restore shipped FP/FN costs (1/10), select min-cost thresholds only on the original calibration split without refitting models, and retain public policy metadata.
- Align Streamlit Local/API effective thresholds and add real-model policy compatibility regression coverage.

- Upgrade scikit-learn to 1.5.2 and regenerate the coherent reference model release from independently verified Kaggle dataset version 3.
- Remove the CVE-2024-5206 exception after a no-exemption installed-environment audit.
- Preserve API schemas and model keys; probabilities, min-cost thresholds, and the policy version intentionally change.
- Record provenance, split fingerprints, calibration/holdout metrics and new golden witnesses; historical performance parity is not claimed.
- See [migration evidence](docs/SKLEARN_MIGRATION.md) for limitations and pending CI/browser gates.

## v0.1.0 — Final release hardening

- Reworked `scripts/train.py` to use separate train, calibration/threshold-selection, and holdout-test reporting splits.
- Marked packaged artifact metrics as reference-only and moved operational metric claims to regenerated artifacts.
- Added configurable CORS allowlisting and prod-mode guards for auth and wildcard CORS.
- Replaced byte-by-byte Redis response parsing with buffered RESP reads.
- Added offset pagination for audit-log reads.
- Made the SQLite store singleton thread-safe.
- Documented policy/artifact cache restart behavior and queue/audit trade-offs.

## v0.1.0 — Release closure fixes

- Routed API and worker scoring through shared application services.
- Added Streamlit API-mode batch chunking based on backend limits.
- Forwarded selected policy presets or manual thresholds from the UI to the API.
- Added expected SHA-256 artifact checksum validation in the readiness contract.
- Expanded Grafana panels for HTTP latency, prediction errors, policy usage, and batch-job status.
- Added regression tests for metadata limits, UI/API chunking, policy forwarding, and checksum-manifest validation.

## v0.1.0 — Final public reference hardening

- Added persisted `/v1/metrics/summary` for reviewer/dashboard operational state.
- Protected `/metadata` when auth mode is enabled.
- Seeded model-version and threshold-policy reference tables from shipped artifacts.
- Moved request/response contracts into `api/schemas.py` for a cleaner API boundary.
- Enabled JSON logs for API and worker services in Docker Compose.
- Added README screenshots using existing `assets/` images.
- Removed the unused PowerShell environment stub.

## v0.1.0

- Added Dependabot configuration matching the documented dependency-update policy.
- Added protected-mode auth forwarding from the Streamlit console to the FastAPI backend.
- Refactored environment settings to resolve values at `Settings()` creation time instead of import time.
- Tightened JWT header validation for the local protected-mode token flow.
- Updated README, API, operations, and security docs to match the implemented auth/dependency behavior.
- Repositioned the project as a production-style ML risk operations reference.
- Added explicit risk decision fields to scoring responses.
- Added fail-closed artifact readiness behavior by default.
- Added policy/version metadata alignment across public artifacts.
- Strengthened audit persistence with policy version, decision, reason hints, and input hashes.
- Tightened policy resolution so invalid explicit policies fail with `400` instead of silently falling back.
- Reported request latency in single-record prediction responses.
- Protected versioned policy/model metadata routes when auth is enabled.
- Rewrote README and architecture docs with confident release-scope wording tied to the implemented system.
- Added engineering notes covering clean-code boundaries and design-pattern choices.
- Removed local runtime/cache files from the distributable release package.
