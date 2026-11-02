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
