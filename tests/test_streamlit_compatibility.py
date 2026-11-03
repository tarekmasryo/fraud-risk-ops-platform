"""Render the real console and exercise model/threshold controls after UI upgrades."""

from streamlit.testing.v1 import AppTest

from fraud_dashboard.core import config
from fraud_dashboard.core.artifacts import project_root


def test_console_renders_and_switches_model_and_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUD_API_URL", "http://127.0.0.1:1")
    monkeypatch.setattr(
        config,
        "_SETTINGS",
        config.Settings(
            database_url=f"sqlite:///{tmp_path / 'ui.db'}",
            allow_local_fallback=True,
            strict_artifact_runtime=True,
            allow_artifact_compatibility_fallback=False,
        ),
    )
    app = AppTest.from_file(str(project_root() / "app.py")).run(timeout=90)
    assert not app.exception, [item.value for item in app.exception]
    assert len(app.tabs) == 5
    app.sidebar.radio[0].set_value("Local")
    app.sidebar.selectbox[0].select("XGBoost (Calibrated)")
    app.sidebar.radio[1].set_value("Manual threshold").run(timeout=90)
    assert not app.exception, [item.value for item in app.exception]
    threshold = next(x for x in app.sidebar.slider if x.label == "Manual decision threshold")
    threshold.set_value(0.42).run(timeout=90)
    assert not app.exception, [item.value for item in app.exception]
    assert app.sidebar.selectbox[0].value == "XGBoost (Calibrated)"
    assert (
        next(x.value for x in app.sidebar.slider if x.label == "Manual decision threshold") == 0.42
    )


def test_console_outage_clears_results_and_retry_recovers_without_fallback(monkeypatch):
    import numpy as np
    import pandas as pd

    from fraud_dashboard.ui import app as ui_app
    from fraud_dashboard.ui.api_client import BackendUnavailableError

    class DemoPredictor:
        last_results = None
        fail_inference = False

        def schema_features(self):
            return ["Time", "Amount"]

        def available_models(self):
            return ["rf"]

        def thresholds(self):
            return {"artifact_version": "test-policy-v1", "RF_Thr_P90": 0.65}

        def predict_proba_batch(self, frame, **kwargs):
            if self.fail_inference:
                raise BackendUnavailableError("Connection dropped during inference")
            return np.array([0.1, 0.9]), 0.01

    p = DemoPredictor()
    state = {"online": True, "bug": False}

    def backend():
        if state["bug"]:
            raise ValueError("Unexpected configuration bug")
        if not state["online"]:
            raise BackendUnavailableError("API unavailable with fallback disabled")
        return p, {}

    monkeypatch.setattr(ui_app, "ui_backend_selector", backend)
    frame = pd.DataFrame({"Time": [0, 1], "Amount": [1, 2], "Class": [0, 1]})
    monkeypatch.setattr(
        ui_app, "get_active_dataframe", lambda upload: (frame.copy(), "Synthetic test")
    )
    monkeypatch.setattr(ui_app.ModelInsightsPage, "render", lambda self: None)
    app = AppTest.from_string("from fraud_dashboard.ui.app import main\nmain()").run(timeout=30)
    assert not app.exception
    assert app.session_state["last_run"].y_prob.tolist() == [0.1, 0.9]
    assert any(
        item.value.columns.tolist()[:2] == ["fraud_proba", "fraud_pred"] for item in app.dataframe
    )
    assert any("Policy version: test-policy-v1" in item.value for item in app.markdown)
    state["online"] = False
    app.run(timeout=30)
    assert not app.exception
    assert "last_run" not in app.session_state
    assert len(app.dataframe) == 0
    assert any("Prediction API unavailable" in item.value for item in app.error)
    assert not app.success
    state["online"] = True
    app.button(key="retry_prediction_api").click().run(timeout=30)
    assert not app.exception
    assert "last_run" in app.session_state
    p.fail_inference = True
    app.run(timeout=30)
    assert not app.exception
    assert "last_run" not in app.session_state
    assert not app.success
    p.fail_inference = False
    app.button(key="retry_prediction_api").click().run(timeout=30)
    assert not app.exception
    assert "last_run" in app.session_state
    state["bug"] = True
    app.run(timeout=30)
    assert app.exception and "Unexpected configuration bug" in app.exception[0].value
