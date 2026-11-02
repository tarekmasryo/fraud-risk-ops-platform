"""Exercise API -> Redis -> separate worker -> SQLite in disposable Compose CI state."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

API = "http://127.0.0.1:8000"
QUEUE = "fraud-risk-ops:jobs"

SNAPSHOT = """
import json, sqlite3, sys
from fraud_dashboard.core.config import get_settings
settings = get_settings()
assert settings.run_jobs_in_api is False, 'RUN_JOBS_IN_API must be false in API'
path = settings.database_url.removeprefix('sqlite:///')
with sqlite3.connect('file:' + path + '?mode=ro', uri=True, timeout=5) as db:
    db.row_factory = sqlite3.Row
    job = dict(db.execute('SELECT * FROM batch_jobs WHERE id = ?', (sys.argv[1],)).fetchone())
    result = json.loads(job.get('result_json') or '{}')
    request_id = result.get('request_id', '')
    request = db.execute('SELECT * FROM prediction_requests WHERE id = ?', (request_id,)).fetchone()
    predictions = [dict(r) for r in db.execute(
        'SELECT * FROM predictions WHERE request_id = ? ORDER BY row_index', (request_id,))]
    audits = [dict(r) for r in db.execute(
        'SELECT * FROM audit_logs WHERE resource_id IN (?, ?)', (sys.argv[1], request_id))]
    print(json.dumps({'job': job, 'request': dict(request) if request else None,
                      'predictions': predictions, 'audits': audits}))
"""

WORKER_CHECK = """
from unittest.mock import patch
from fraud_dashboard.core.config import get_settings
from fraud_dashboard.platform.store import OpsStore
from fraud_dashboard.workers import worker
assert get_settings().run_jobs_in_api is False, 'RUN_JOBS_IN_API must be false in worker'
with patch.object(OpsStore, 'claim_next_job', side_effect=AssertionError('SQLite polling forbidden')):
    with patch.object(worker, 'dequeue_job', return_value=None):
        try:
            worker.run_once()
        except AssertionError as exc:
            assert str(exc) == 'SQLite polling forbidden'
            print('PASS: SQLite-only processing rejected by test guard', flush=True)
        else:
            raise AssertionError('SQLite fallback guard was not exercised')
    assert worker.run_once(), 'Redis worker did not consume a job'
    print('PASS: separate worker processed a real Redis wake-up without SQLite polling', flush=True)
"""


def compose(*args: str) -> str:
    result = subprocess.run(
        ["docker", "compose", *args], capture_output=True, text=True, timeout=90, check=True
    )
    return result.stdout.strip()


def api(path: str, payload: dict | None = None) -> dict:
    request = Request(
        API + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def redis_jobs() -> list[str]:
    return compose(
        "exec", "-T", "redis", "redis-cli", "--raw", "LRANGE", QUEUE, "0", "-1"
    ).splitlines()


def snapshot(job_id: str) -> dict:
    return json.loads(compose("exec", "-T", "api", "python", "-c", SNAPSHOT, job_id))


def verify() -> None:
    compose("stop", "worker")
    assert not compose("ps", "--status", "running", "--quiet", "worker"), "Worker is still running"
    assert redis_jobs() == [], "Expected a fresh disposable CI queue"
    settings_check = (
        "from fraud_dashboard.core.config import get_settings; "
        "assert get_settings().run_jobs_in_api is False, 'RUN_JOBS_IN_API must be false'"
    )
    compose("exec", "-T", "api", "python", "-c", settings_check)
    records = json.loads(
        (
            Path(__file__).resolve().parents[1] / "tests/fixtures/reference_predictions.json"
        ).read_text()
    )["records"][:2]
    created = api("/v1/batch-jobs", {"records": records, "model": "rf", "policy": "balanced"})
    job_id = created["job_id"]
    assert created["status"] == "queued", created
    assert api(f"/v1/jobs/{job_id}")["status"] == "queued", "API executed the queued job"
    before = snapshot(job_id)
    assert before["job"]["status"] == "queued", before
    assert json.loads(before["job"]["request_json"])["records"] == records, before
    assert redis_jobs() == [job_id], f"Redis must contain exactly the persisted job {job_id}"
    print(
        f"PASS: RUN_JOBS_IN_API=false; worker stopped; {job_id} persisted queued and in Redis",
        flush=True,
    )

    # A one-shot process uses the existing worker service image, environment and volume.
    # The production worker remains unchanged; only this process forbids SQLite polling.
    print(
        compose("run", "--rm", "--no-deps", "-T", "worker", "python", "-c", WORKER_CHECK),
        flush=True,
    )
    deadline = time.monotonic() + 30
    while True:
        job = api(f"/v1/jobs/{job_id}")
        if job["status"] == "completed":
            break
        assert job["status"] != "failed", job
        assert time.monotonic() < deadline, f"Timed out waiting for {job_id}: {job}"
        time.sleep(1)
    assert redis_jobs() == [], f"Redis wake-up for {job_id} was not consumed"
    after = snapshot(job_id)
    persisted = after["job"]
    assert persisted["status"] == "completed" and not persisted["error"], after
    assert persisted["total_records"] == persisted["processed_records"] == len(records), after
    assert persisted["failed_records"] == 0, after
    request = after["request"]
    assert request["model_key"] == "rf", after
    assert request["source"] == "async_job" and request["status"] == "completed", after
    assert request["record_count"] == len(records), after
    assert len(after["predictions"]) == len(records), after
    assert [p["row_index"] for p in after["predictions"]] == list(range(len(records))), after
    result = json.loads(persisted["result_json"])
    assert len(result["results"]) == len(records), after
    for prediction, response in zip(after["predictions"], result["results"], strict=True):
        assert prediction["proba_fraud"] == response["proba_fraud"], after
        assert prediction["input_hash"] == response["input_hash"], after
    audits = {(a["action"], a["resource_type"], a["resource_id"]) for a in after["audits"]}
    assert ("batch_job_created", "batch_job", job_id) in audits, after
    assert ("prediction_requested", "prediction_request", request["id"]) in audits, after
    print(
        f"PASS: {job_id} consumed from Redis and completed by separate worker; "
        f"request={request['id']}; {len(records)} predictions and both audit records persisted",
        flush=True,
    )


def main() -> int:
    if os.environ.get("GITHUB_ACTIONS") != "true" or not os.environ.get(
        "COMPOSE_PROJECT_NAME", ""
    ).startswith("fraud-risk-ops-ci-"):
        print("Run only against the isolated GitHub Actions Compose project", file=sys.stderr)
        return 1
    try:
        verify()
    except Exception as exc:
        print(f"Redis/worker integration failed: {exc}", file=sys.stderr)
        if isinstance(exc, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            print(exc.stdout, exc.stderr, file=sys.stderr)
        for args in [
            ("ps", "--all"),
            ("logs", "--no-color", "--tail", "80", "api", "worker", "redis"),
        ]:
            try:
                print(compose(*args), file=sys.stderr)
            except Exception as diagnostic_error:
                print(f"Diagnostics unavailable: {diagnostic_error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
