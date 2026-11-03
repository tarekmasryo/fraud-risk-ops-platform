from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud_dashboard.api import main as api_main
from fraud_dashboard.api.main import app
from fraud_dashboard.core.artifact_contract import artifact_checksums, check_artifact_contract
from fraud_dashboard.core.policy import get_policy_threshold, load_policy


def _record() -> dict[str, float]:
    meta = json.loads(
        (Path(__file__).resolve().parents[1] / "artifacts" / "metadata.json").read_text()
    )
    return {feature: 0.0 for feature in meta["schema"]["features"]}


class _FakeModel:
    def predict_proba(self, X):
        scores = np.full(len(X), 0.1, dtype=float)
        return np.column_stack([1.0 - scores, scores])


def _sample_bundle() -> dict:
    features = list(_record().keys())
    policy = load_policy()
    return {
        "available_models": ["rf", "xgb"],
        "metadata": {"env": {}, "models": {}},
        "schema": {"features": features},
        "thresholds": {"default_model": "rf", "models": {"rf": {"threshold": 0.2}}},
        "policy": policy,
        "models": {"rf": _FakeModel(), "xgb": _FakeModel()},
    }


@pytest.fixture()
def patched_client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    from fraud_dashboard.core import config
    from fraud_dashboard.platform import store

    monkeypatch.setattr(api_main, "_bundle", _sample_bundle)
    monkeypatch.setattr(api_main, "_policy", load_policy)
    config._SETTINGS = config.Settings(database_url=f"sqlite:///{tmp_path / 'ops.db'}")  # type: ignore[misc]
    store._STORE = None  # type: ignore[misc]

    with TestClient(app) as client:
        yield client

    store._STORE = None  # type: ignore[misc]
    config._SETTINGS = None


def test_policy_file_is_single_source_of_truth() -> None:
    policy = load_policy()
    assert policy["default_model"] == "rf"
    assert (
        get_policy_threshold(policy, model_key="rf", policy_name="min_cost")
        == 0.0016451652802712843
    )
    assert get_policy_threshold(policy, model_key="xgb", policy_name="balanced") == 0.75


def test_artifact_contract_shape() -> None:
    result = check_artifact_contract(run_sample_prediction=False)
    assert result.checks["metadata_present"]
    assert result.checks["schema_features_present"]
    assert result.checks["policy_present"]
    sums = artifact_checksums()
    assert "metadata.json" in sums
    assert "policy.json" in sums


def test_live_ready_metadata_and_metrics_endpoints() -> None:
    with TestClient(app) as client:
        assert client.get("/live").json()["status"] == "alive"
        ready = client.get("/ready")
        assert ready.status_code in {200, 503}
        meta = client.get("/metadata")
        assert meta.status_code == 200
        assert "policy" in meta.json()
        metrics = client.get("/metrics")
        assert metrics.status_code == 200


def test_v1_prediction_and_audit_log(patched_client: TestClient) -> None:
    r = patched_client.post(
        "/v1/predictions", json={"record": _record(), "model": "rf", "policy": "min_cost"}
    )
    assert r.status_code == 200
    out = r.json()
    assert out["request_id"]
    logs = patched_client.get("/v1/audit-logs")
    assert logs.status_code == 200
    assert isinstance(logs.json(), list)


def test_unknown_named_policy_fails_closed(patched_client: TestClient) -> None:
    r = patched_client.post(
        "/v1/predictions",
        json={"record": _record(), "model": "rf", "policy": "does_not_exist"},
    )
    assert r.status_code == 400
    assert "Unknown policy" in r.text


def test_prediction_latency_is_reported(patched_client: TestClient) -> None:
    r = patched_client.post(
        "/v1/predictions", json={"record": _record(), "model": "rf", "policy": "min_cost"}
    )
    assert r.status_code == 200
    assert isinstance(r.json()["latency_ms"], int)


def test_batch_limit_rejects_oversized_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from fraud_dashboard.core import config

    config._SETTINGS = config.Settings(max_batch_records=1)  # type: ignore[misc]
    with TestClient(app) as client:
        r = client.post(
            "/v1/predictions/batch", json={"records": [_record(), _record()], "model": "rf"}
        )
        assert r.status_code == 422
    config._SETTINGS = None


def test_ops_store_job_claiming(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fraud_dashboard.core import config
    from fraud_dashboard.platform.store import OpsStore

    db_path = tmp_path / "ops.db"
    config._SETTINGS = config.Settings(database_url=f"sqlite:///{db_path}")  # type: ignore[misc]
    store = OpsStore(database_url=f"sqlite:///{db_path}")
    job_id = store.create_job(
        model_key="rf",
        threshold=0.5,
        total_records=1,
        request={"records": [{"Amount": 0.0}], "model": "rf"},
    )

    claimed = store.claim_job(job_id)
    assert claimed is not None
    assert claimed["id"] == job_id
    assert claimed["status"] == "running"
    assert claimed["request"]["model"] == "rf"
    assert store.claim_job(job_id) is None
    config._SETTINGS = None


def test_artifact_checksum_manifest_is_validated() -> None:
    result = check_artifact_contract(run_sample_prediction=False)
    assert result.checks["artifact_checksum_manifest_present"]
    assert result.checks["artifact_checksums_match_manifest"]
