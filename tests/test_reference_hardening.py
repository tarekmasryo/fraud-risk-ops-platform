"""Regression checks for the bounded reference hardening (synthetic inputs only)."""

import io
import json
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from fraud_dashboard.api import main as api
from fraud_dashboard.api.schemas import PredictBatchRequest
from fraud_dashboard.core import artifact_contract, artifacts, config
from fraud_dashboard.core.decision import reason_codes_for_record
from fraud_dashboard.core.errors import ArtifactContractError
from fraud_dashboard.platform import store
from fraud_dashboard.services import scoring
from fraud_dashboard.services.batch_jobs import run_batch_job
from fraud_dashboard.ui import data_io, pages
from tests.test_api import _load_demo_record

ROOT = artifacts.project_root()


@pytest.fixture
def invalid_release(tmp_path, monkeypatch):
    # Small, deliberately invalid bytes: no real models are copied or altered.
    adir = tmp_path / "artifacts"
    adir.mkdir()
    for name in ("metadata.json", "policy.json", "thresholds.json"):
        (adir / name).write_bytes((ROOT / "artifacts" / name).read_bytes())
    for name in ("rf_calibrated.joblib", "xgb_calibrated.joblib"):
        (adir / name).write_bytes(b"not a serialized model")
    monkeypatch.setattr(
        config,
        "_SETTINGS",
        config.Settings(
            model_artifact_dir=str(adir),
            database_url=f"sqlite:///{tmp_path / 'test.db'}",
            strict_artifact_runtime=False,
            allow_artifact_compatibility_fallback=True,
        ),
    )
    monkeypatch.setattr(store, "_STORE", None)
    api._bundle.cache_clear()
    api._policy.cache_clear()
    scoring._validated_bundle.cache_clear()
    yield adir
    api._bundle.cache_clear()
    api._policy.cache_clear()
    scoring._validated_bundle.cache_clear()


@pytest.mark.parametrize(
    "failure",
    [
        "checksum",
        "missing_manifest",
        "missing_entry",
        "runtime",
        "missing_runtime",
        "malformed_metadata",
        "missing_file",
    ],
)
def test_invalid_artifacts_never_deserialize(invalid_release, monkeypatch, failure):
    path = invalid_release / "metadata.json"
    meta = json.loads(path.read_text())
    if failure == "missing_manifest":
        meta.pop("artifact_integrity")
    elif failure == "missing_entry":
        meta["artifact_integrity"]["expected_sha256"].pop("rf_calibrated.joblib")
    elif failure == "runtime":
        meta["env"]["scikit_learn"] = "0.0.0"
    elif failure == "missing_runtime":
        meta["env"] = {}
    elif failure == "missing_file":
        (invalid_release / "rf_calibrated.joblib").unlink()
    path.write_text("{" if failure == "malformed_metadata" else json.dumps(meta))
    loader = Mock(side_effect=AssertionError("Deserialization must not happen"))
    monkeypatch.setattr(artifacts.joblib, "load", loader)
    with pytest.raises(ArtifactContractError):
        artifacts.load_bundle()
    loader.assert_not_called()


def test_readiness_does_not_load_on_checksum_failure(invalid_release, monkeypatch):
    loader = Mock(side_effect=AssertionError("Sample must not load"))
    monkeypatch.setattr(artifact_contract, "load_models", loader)
    result = artifact_contract.check_artifact_contract()
    assert not result.ok
    assert not result.checks["sample_prediction"]
    loader.assert_not_called()


def test_api_and_worker_fail_closed(invalid_release, monkeypatch):
    loader = Mock(side_effect=AssertionError("Invalid model must never load"))
    monkeypatch.setattr(artifacts.joblib, "load", loader)
    record = _load_demo_record()
    with TestClient(api.app) as client:
        response = client.post("/v1/predictions", json={"record": record, "model": "rf"})
        assert response.status_code == 503
        assert "failed validation" in response.json()["detail"]
    ops = store.get_store()
    req = PredictBatchRequest(records=[record], model="rf")
    job_id = ops.create_job(
        model_key="rf", threshold=0.5, total_records=1, request=req.model_dump()
    )
    run_batch_job(job_id, req)
    job = ops.get_job(job_id)
    assert job["status"] == "failed"
    assert job["processed_records"] == 0
    assert "ArtifactContractError" in job["error"]
    loader.assert_not_called()


def test_default_service_reuses_validated_bundle(monkeypatch):
    scoring._validated_bundle.cache_clear()
    bundle = {"policy": {"policy_version": "test"}}
    loader = Mock(return_value=bundle)
    monkeypatch.setattr(scoring, "load_bundle", loader)
    try:
        for _ in range(3):
            service = scoring.FraudScoringService()
            assert service.bundle() is bundle
            assert service.policy() is bundle["policy"]
        loader.assert_called_once()
    finally:
        scoring._validated_bundle.cache_clear()


@pytest.fixture
def ui(monkeypatch):
    from fraud_dashboard.ui.theme import init_plotly_theme

    init_plotly_theme()
    ui = MagicMock()
    ui.session_state = {}
    ui.columns.side_effect = lambda count: [MagicMock() for _ in range(count)]
    ui.button.return_value = False
    ui.checkbox.return_value = False
    monkeypatch.setattr(pages, "st", ui)
    monkeypatch.setattr(data_io, "st", ui)
    monkeypatch.setattr(pages, "plotly_full_width", Mock())
    return ui


def frame():
    return pd.DataFrame(
        {
            "Time": [0.0, 3600.0, 86400.0, 90000.0],
            "Amount": [1.0, 2.0, 3.0, 4.0],
            "Class": [0, 1, 0, 1],
        }
    )


def predict(df, ui):
    predictor = Mock()
    predictor.predict_proba_batch.return_value = (np.array([0.1, 0.9, 0.2, 0.8]), 0.01)
    pages.PredictionEnginePage(predictor=predictor, expected_features=["Time", "Amount"]).render(
        df_pred=df,
        src_msg="Synthetic test",
        model_key="rf",
        thr=0.5,
        policy_name=None,
        cost_fp=5.0,
        cost_fn=200.0,
    )
    return predictor


@pytest.mark.parametrize("kind", ["same", "different_length", "same_length", "reordered"])
def test_profile_only_uses_predictions_for_identical_dataset(ui, kind):
    original = frame()
    predict(original, ui)
    profile = original.copy()
    if kind == "different_length":
        profile = profile.iloc[:2]
    elif kind == "same_length":
        profile.loc[0, "Amount"] = 999.0
    elif kind == "reordered":
        profile = profile.iloc[::-1].reset_index(drop=True)
    ui.markdown.reset_mock()
    pages.DataQualitySegmentsPage().render(df=profile, src_msg="Test", thr=0.5)
    segmented = any("Segmented Metrics" in str(call) for call in ui.markdown.call_args_list)
    assert segmented == (kind == "same")
    assert not ui.error.called


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"Time,Amount\n",
        b'Time,Amount\n0,"unterminated\n',
        b"Time,Amount\n0,invalid\n",
        b"Time,Amount\n0,inf\n",
    ],
)
def test_bad_csv_upload_is_actionable_and_empty(ui, payload):
    result, message = data_io.get_active_dataframe(io.BytesIO(payload))
    assert result.empty
    assert message == "Upload failed."
    assert "Invalid CSV" in ui.error.call_args.args[0]


def test_valid_csv_upload(ui):
    result, _ = data_io.get_active_dataframe(io.BytesIO(frame().to_csv(index=False).encode()))
    pd.testing.assert_frame_equal(result, frame())
    assert not ui.error.called


@pytest.mark.parametrize("kind", ["missing", "nonnumeric", "empty", "invalid_label"])
def test_prediction_rejects_invalid_input_and_clears_stale_run(ui, kind):
    df = frame()
    predict(df, ui)
    assert "last_run" in ui.session_state
    if kind == "missing":
        df = df.drop(columns="Time")
    elif kind == "nonnumeric":
        df = df.astype({"Amount": object})
        df.loc[0, "Amount"] = "bad"
    elif kind == "invalid_label":
        df.loc[0, "Class"] = 7
    else:
        df = df.iloc[:0]
    predictor = predict(df, ui)
    predictor.predict_proba_batch.assert_not_called()
    assert "last_run" not in ui.session_state
    assert ui.error.called or ui.warning.called


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_skips_roc_and_importance(ui, monkeypatch, label):
    df = frame().assign(Class=label)
    predict(df, ui)
    roc = Mock(side_effect=AssertionError("ROC is undefined"))
    importance = Mock(side_effect=AssertionError("AUC importance is undefined"))
    monkeypatch.setattr(pages, "roc_curve", roc)
    monkeypatch.setattr(pages, "permutation_importance", importance)
    metrics_page().render(thr=0.5, cost_fp=5.0, cost_fn=200.0)
    pages.ModelInsightsPage(top_k=2).render()
    roc.assert_not_called()
    importance.assert_not_called()
    assert any("both Class=0 and Class=1" in str(call) for call in ui.info.call_args_list)


def metrics_page():
    return pages.ModelMetricsPage(
        target_precision=0.9,
        target_recall=0.8,
        fp_warn=10,
        thr_p90_default=0.5,
        thr_mincost_default=0.2,
    )


def test_two_class_evaluation_still_runs(ui, monkeypatch):
    predict(frame(), ui)
    roc = Mock(wraps=pages.roc_curve)
    monkeypatch.setattr(pages, "roc_curve", roc)
    metrics_page().render(thr=0.5, cost_fp=5.0, cost_fn=200.0)
    roc.assert_called_once()
    monkeypatch.setattr(pages.ModelInsightsPage, "_models", lambda self: {"rf": object()})
    importance = Mock(
        return_value=Mock(
            importances_mean=np.array([0.1, 0.2]), importances_std=np.array([0.0, 0.0])
        )
    )
    monkeypatch.setattr(pages, "permutation_importance", importance)
    pages.ModelInsightsPage(top_k=2).render()
    importance.assert_called_once()


def test_elapsed_time_does_not_create_wall_clock_risk():
    results = [
        reason_codes_for_record({"Time": t, "Amount": 1000}, score=0.8, threshold=0.5)
        for t in [0, 3600, 43200, 86400, 172800]
    ]
    assert all(codes == results[0] for codes in results)
    assert "amount_high" in results[0]
    assert "score_above_policy_threshold" in results[0]
    assert "unusual_time_window" not in results[0]
    engineered = pages.build_engineered_features(frame())
    assert engineered["Hour_from_start_mod24"].tolist() == [0, 1, 0, 1]
    assert "is_business_hours_proxy" not in engineered
    assert "is_night_proxy" not in engineered


def test_compose_localhost_and_build_context_allowlist():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    for service in compose["services"].values():
        for port in service.get("ports", []):
            assert port.startswith("127.0.0.1:")


def test_readiness_reports_malformed_metadata_without_loading(invalid_release, monkeypatch):
    (invalid_release / "metadata.json").write_text("{")
    loader = Mock(side_effect=AssertionError("Must not deserialize"))
    monkeypatch.setattr(artifact_contract, "load_models", loader)
    result = artifact_contract.check_artifact_contract()
    assert not result.ok
    assert "Cannot read artifact metadata" in result.errors[0]
    loader.assert_not_called()


def test_single_class_importance_sample_is_rejected(ui, monkeypatch):
    predict(frame(), ui)
    state = ui.session_state["last_run"]
    state.X = pd.DataFrame({"Time": np.arange(6001), "Amount": np.ones(6001)})
    state.y_true = np.zeros(6001)
    selected = np.random.RandomState(42).choice(6001, 6000, replace=False)
    omitted = next(iter(set(range(6001)) - set(selected)))
    state.y_true[omitted] = 1
    monkeypatch.setattr(pages.ModelInsightsPage, "_models", lambda self: {"rf": object()})
    importance = Mock(side_effect=AssertionError("Single-class sample cannot be evaluated"))
    monkeypatch.setattr(pages, "permutation_importance", importance)
    pages.ModelInsightsPage(top_k=2).render()
    importance.assert_not_called()
    assert "sample contains only one class" in ui.info.call_args.args[0]


def _render_csv_component(payload):
    import io

    import streamlit as st

    from fraud_dashboard.ui.data_io import get_active_dataframe
    from fraud_dashboard.ui.pages import DataOverviewPage, DataQualitySegmentsPage
    from fraud_dashboard.ui.theme import init_plotly_theme

    init_plotly_theme()
    df, message = get_active_dataframe(io.BytesIO(payload))
    DataOverviewPage(expected_features=["Time", "Amount"]).render(df=df, src_msg=message)
    DataQualitySegmentsPage().render(df=df, src_msg=message, thr=0.5)
    st.write("Finished rendering")


@pytest.mark.parametrize(
    "payload",
    [b"Time,Amount\n0,1\n", b"Time,Amount\n0,bad\n", b'Time,Amount\n0,"unterminated\n', b""],
)
def test_csv_component_renders_without_tracebacks(payload):
    # Component evidence only: this does not automate a browser or file-upload control.
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_csv_component, args=(payload,)).run(timeout=30)
    assert not app.exception
    assert bool(app.error) == (payload != b"Time,Amount\n0,1\n")
    assert any("Finished rendering" in item.value for item in app.markdown)
