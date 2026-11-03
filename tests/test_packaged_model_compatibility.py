"""Golden and integrity regression evidence for the migrated reference release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_dashboard.core.artifacts import load_bundle, project_root


def _fitted_objects(value):
    pending = [value]
    seen = set()
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, (list, tuple)):
            pending.extend(item)
        elif isinstance(item, np.ndarray) and item.dtype == object:
            pending.extend(item.flat)
        elif hasattr(item, "__dict__") and not isinstance(item, type):
            pending.extend(vars(item).values())


def test_packaged_predictions_and_numeric_model_graph():
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "reference_predictions.json").read_text()
    )
    for name, expected in reference["sha256"].items():
        assert (
            hashlib.sha256((project_root() / "artifacts" / name).read_bytes()).hexdigest()
            == expected
        )
    bundle = load_bundle()
    assert bundle["schema"]["features"] == reference["features"]
    frame = pd.DataFrame(reference["records"], columns=reference["features"])
    assert set(bundle["models"]) == set(reference["probabilities"])
    for family, model in bundle["models"].items():
        for fitted in _fitted_objects(model):
            assert not type(fitted).__module__.startswith("sklearn.feature_extraction.text"), (
                "The numeric reference model contains unexpected text feature extraction"
            )
        actual = model.predict_proba(frame)[:, 1]
        assert np.isfinite(actual).all()
        assert ((actual >= 0) & (actual <= 1)).all()
        np.testing.assert_allclose(actual, reference["probabilities"][family], rtol=1e-7, atol=1e-9)


def test_migrated_release_api_and_worker_persist_real_model_results(monkeypatch, tmp_path):
    import sqlite3

    from fastapi.testclient import TestClient

    from fraud_dashboard.api import main as api
    from fraud_dashboard.api.schemas import PredictBatchRequest
    from fraud_dashboard.core import config
    from fraud_dashboard.platform import store
    from fraud_dashboard.services import scoring
    from fraud_dashboard.services.batch_jobs import run_batch_job

    reference = json.loads(
        (Path(__file__).parent / "fixtures/reference_predictions.json").read_text()
    )
    database = tmp_path / "migrated-release.db"
    monkeypatch.setattr(
        config,
        "_SETTINGS",
        config.Settings(
            database_url=f"sqlite:///{database}",
            model_artifact_dir=str(project_root() / "artifacts"),
            strict_artifact_runtime=True,
            allow_artifact_compatibility_fallback=False,
        ),
    )
    monkeypatch.setattr(store, "_STORE", None)
    api._bundle.cache_clear()
    api._policy.cache_clear()
    scoring._validated_bundle.cache_clear()
    try:
        with TestClient(api.app) as client:
            assert client.get("/ready").status_code == 200
            for family, probabilities in reference["probabilities"].items():
                for settings in [
                    {"policy": "balanced"},
                    {"policy": "min_cost"},
                    {"threshold": 0.0},
                    {"threshold": 1.0},
                ]:
                    response = client.post(
                        "/v1/predictions/batch",
                        json={"records": reference["records"], "model": family, **settings},
                    )
                    assert response.status_code == 200, response.text
                    result = response.json()
                    np.testing.assert_allclose(
                        [item["proba_fraud"] for item in result["results"]],
                        probabilities,
                        rtol=1e-7,
                        atol=1e-9,
                    )
                    assert [item["label"] for item in result["results"]] == [
                        int(p >= result["threshold"]) for p in probabilities
                    ]
                request = PredictBatchRequest(
                    records=reference["records"], model=family, policy="min_cost"
                )
                ops = store.get_store()
                job_id = ops.create_job(
                    model_key=family, threshold=0.5, total_records=4, request=request.model_dump()
                )
                assert ops.claim_job(job_id)["status"] == "running"
                run_batch_job(job_id, request)
                job = ops.get_job(job_id)
                assert job["status"] == "completed" and job["processed_records"] == 4
                np.testing.assert_allclose(
                    [item["proba_fraud"] for item in job["result"]["results"]],
                    probabilities,
                    rtol=1e-7,
                    atol=1e-9,
                )
        with sqlite3.connect(database) as db:
            assert db.execute("SELECT count(*) FROM predictions").fetchone()[0] == 40
            assert db.execute("SELECT count(*) FROM prediction_requests").fetchone()[0] == 10
            assert db.execute("SELECT count(*) FROM audit_logs").fetchone()[0] >= 10
    finally:
        api._bundle.cache_clear()
        api._policy.cache_clear()
        scoring._validated_bundle.cache_clear()
