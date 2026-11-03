from __future__ import annotations

import pandas as pd

from fraud_dashboard.ui import predictors
from fraud_dashboard.ui.api_client import ApiConfig
from fraud_dashboard.ui.predictors import ApiPredictor


def test_api_predictor_chunks_large_batches_and_forwards_policy(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_fetch_metadata(_cfg):
        return {"limits": {"max_batch_records": 2}}

    def fake_predict_batch(*, cfg, model, records, threshold=None, policy=None):
        calls.append(
            {
                "cfg": cfg,
                "model": model,
                "records": records,
                "threshold": threshold,
                "policy": policy,
            }
        )
        return {"results": [{"proba_fraud": 0.25} for _ in records]}

    monkeypatch.setattr(predictors, "fetch_metadata", fake_fetch_metadata)
    monkeypatch.setattr(predictors, "predict_batch", fake_predict_batch)

    predictor = ApiPredictor(ApiConfig(base_url="http://api:8000"))
    df = pd.DataFrame({"Amount": [1, 2, 3, 4, 5]})

    probs, _elapsed = predictor.predict_proba_batch(df, model_key="rf", policy="balanced")

    assert probs.tolist() == [0.25] * 5
    assert [len(call["records"]) for call in calls] == [2, 2, 1]
    assert {call["policy"] for call in calls} == {"balanced"}
    assert {call["threshold"] for call in calls} == {None}
    assert {call["model"] for call in calls} == {"rf"}


def test_api_predictor_forwards_manual_threshold(monkeypatch) -> None:
    calls: list[dict] = []

    monkeypatch.setattr(
        predictors,
        "fetch_metadata",
        lambda _cfg: {"limits": {"max_batch_records": 1000}},
    )

    def fake_predict_batch(*, cfg, model, records, threshold=None, policy=None):
        calls.append({"threshold": threshold, "policy": policy, "records": records})
        return {"results": [{"proba_fraud": 0.7} for _ in records]}

    monkeypatch.setattr(predictors, "predict_batch", fake_predict_batch)

    predictor = ApiPredictor(ApiConfig(base_url="http://api:8000"))
    df = pd.DataFrame({"Amount": [10, 20]})

    probs, _elapsed = predictor.predict_proba_batch(df, model_key="xgb", threshold=0.42)

    assert probs.tolist() == [0.7, 0.7]
    assert calls == [
        {
            "threshold": 0.42,
            "policy": None,
            "records": [{"Amount": 10}, {"Amount": 20}],
        }
    ]


def test_local_and_api_share_effective_policies_and_public_contract(monkeypatch, tmp_path):
    import json

    import numpy as np
    from fastapi.testclient import TestClient

    from fraud_dashboard.api import main as api
    from fraud_dashboard.core import config
    from fraud_dashboard.core.artifacts import load_bundle, project_root
    from fraud_dashboard.platform import store
    from fraud_dashboard.ui.app import _policy_threshold, _threshold_defaults
    from fraud_dashboard.ui.predictors import LocalPredictor

    monkeypatch.setattr(
        config,
        "_SETTINGS",
        config.Settings(
            database_url=f"sqlite:///{tmp_path / 'policy.db'}",
            strict_artifact_runtime=True,
            allow_artifact_compatibility_fallback=False,
        ),
    )
    monkeypatch.setattr(store, "_STORE", None)
    monkeypatch.setattr(predictors, "_local_bundle", load_bundle)
    api._bundle.cache_clear()
    api._policy.cache_clear()
    fixture = json.loads((project_root() / "tests/fixtures/reference_predictions.json").read_text())
    frame = pd.DataFrame(fixture["records"], columns=fixture["features"])
    expected_contract = {
        "approve": "risk_score below the selected review threshold",
        "review": "risk_score at or above the selected review threshold",
        "block": "reserved for downstream case-management integration",
    }
    expected_bands = [
        {"name": name, "min": low, "max": high}
        for name, low, high in [
            ("low", 0.0, 0.2),
            ("medium", 0.2, 0.65),
            ("high", 0.65, 0.9),
            ("critical", 0.9, 1.0),
        ]
    ]
    try:
        with TestClient(api.app) as client:
            policy = client.get("/v1/policies").json()
            assert policy["decision_contract"] == expected_contract
            assert policy["risk_bands"] == expected_bands
            assert (
                policy["metrics_boundary"]
                == "Packaged thresholds are suitable for demo/review workflows; operational metrics should be regenerated on the target dataset with a separate holdout test split."
            )
            assert policy["costs"] == {"false_positive": 1.0, "false_negative": 10.0}
            monkeypatch.setattr(
                predictors, "fetch_metadata", lambda cfg: client.get("/metadata").json()
            )
            local = LocalPredictor()
            remote = ApiPredictor(ApiConfig(base_url="http://testserver"))
            assert local.thresholds() == remote.thresholds()
            assert local.thresholds()["COST_FP"] == 1 and local.thresholds()["COST_FN"] == 10
            for family in ["rf", "xgb"]:
                balanced, minimum = _threshold_defaults(local.thresholds(), family)
                assert balanced == policy["policies"]["balanced"]["thresholds"][family]
                assert minimum == policy["policies"]["min_cost"]["thresholds"][family]
                assert local.thresholds()[family.upper() + "_Thr_MinCost"] == minimum
                probs, _ = local.predict_proba_batch(frame, model_key=family)
                for name, cfg in policy["policies"].items():
                    resolved = _policy_threshold(local.thresholds(), name, family, -1.0)
                    assert resolved == cfg["thresholds"][family]
                    response = client.post(
                        "/v1/predictions/batch",
                        json={"records": fixture["records"], "model": family, "policy": name},
                    )
                    assert response.status_code == 200, response.text
                    body = response.json()
                    assert body["threshold"] == resolved
                    np.testing.assert_allclose(
                        [r["proba_fraud"] for r in body["results"]], probs, rtol=1e-7, atol=1e-9
                    )
                    assert [r["label"] for r in body["results"]] == (probs >= resolved).astype(
                        int
                    ).tolist()
    finally:
        api._bundle.cache_clear()
        api._policy.cache_clear()
