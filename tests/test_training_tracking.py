from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from scripts import train as training

RELEASE_FILES = {
    "rf_pipe.joblib",
    "rf_calibrated.joblib",
    "xgb_pipe.joblib",
    "xgb_calibrated.joblib",
    "thresholds.json",
    "policy.json",
    "metadata.json",
}
METRICS = {
    "threshold",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "review_rate",
    "total_cost",
    "average_cost",
    "tp",
    "fp",
    "tn",
    "fn",
}


class _FixedDatetime:
    @staticmethod
    def utcnow():
        return datetime(2026, 1, 1)


def _train_kwargs(tmp_path):
    return {
        "data_path": tmp_path / "input.csv",
        "out_dir": tmp_path / "release",
        "label_col": "Class",
        "seed": 42,
        "calibration_size": 0.2,
        "test_size": 0.2,
        "cost_fp": 5.0,
        "cost_fn": 500.0,
    }


@pytest.mark.parametrize("register_models", [False, True])
def test_real_tracking_preserves_training_and_artifact_contract(
    tmp_path, monkeypatch, register_models
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(training, "datetime", _FixedDatetime)
    kwargs = _train_kwargs(tmp_path)
    # Balanced, separable synthetic data keeps the real estimators fast and deterministic.
    frame = pd.DataFrame({"V1": [0, 1] * 100, "Amount": [10.0, 20.0] * 100, "Class": [0, 1] * 100})
    frame.to_csv(kwargs["data_path"], index=False)
    original_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    try:
        training.train(**kwargs, tracking_enabled=False)
        baseline = {name: (kwargs["out_dir"] / name).read_bytes() for name in RELEASE_FILES}
        assert not (tmp_path / "mlflow.db").exists()
        # A reused output directory must not upload unrelated or stale files.
        (kwargs["out_dir"] / "private.txt").write_text("not a release artifact")
        with ExitStack() as stack:
            spies = {
                name: stack.enter_context(patch.object(mlflow, name, wraps=getattr(mlflow, name)))
                for name in ("log_params", "log_metrics", "log_artifact")
            }
            training.train(**kwargs, tracking_uri=uri, register_models=register_models)
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        registered = client.search_registered_models()
        assert {model.name for model in registered} == (
            {"fraud-risk-rf", "fraud-risk-xgb"} if register_models else set()
        )
        assert all(not model.aliases for model in registered)
        experiment = client.get_experiment_by_name("fraud-risk-training")
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        assert run.info.status == "FINISHED"
        assert "mlflow.parentRunId" not in run.data.tags
        assert mlflow.active_run() is None
        assert spies["log_params"].call_count == 3
        assert spies["log_metrics"].call_count == 4
        assert spies["log_artifact"].call_count == 7

        expected_params = {
            "seed": "42",
            "calibration_size": "0.2",
            "test_size": "0.2",
            "cost_fp": "5.0",
            "cost_fn": "500.0",
            "label_col": "Class",
            "dataset_name": "input.csv",
            "dataset_sha256": training._sha256_file(kwargs["data_path"]),
            "train_rows": "120",
            "calibration_rows": "40",
            "test_rows": "40",
            "feature_count": "2",
            "rf.n_estimators": "300",
            "rf.class_weight": "balanced_subsample",
            "rf.random_state": "42",
            "rf.n_jobs": "-1",
            "xgb.n_estimators": "600",
            "xgb.max_depth": "4",
            "xgb.learning_rate": "0.05",
            "xgb.subsample": "0.8",
            "xgb.colsample_bytree": "0.8",
            "xgb.reg_lambda": "1.0",
            "xgb.random_state": "42",
            "xgb.scale_pos_weight": "1.0",
        }
        assert expected_params.items() <= run.data.params.items()
        assert str(tmp_path) not in str(run.data.params)
        thresholds = json.loads((kwargs["out_dir"] / "thresholds.json").read_text())
        expected_metrics = {}
        for model in ("rf", "xgb"):
            for split, key in (
                ("calibration", "threshold_selection"),
                ("holdout_test", "holdout_evaluation"),
            ):
                evaluation = thresholds["models"][model][key]
                assert set(evaluation) == METRICS
                expected_metrics.update(
                    {f"{model}.{split}.{name}": value for name, value in evaluation.items()}
                )
        assert run.data.metrics == pytest.approx(expected_metrics)
        logged = client.list_artifacts(run.info.run_id, "release")
        assert {Path(item.path).name for item in logged} == RELEASE_FILES
        assert (tmp_path / "mlflow.db").is_file()
        assert (tmp_path / "mlruns").is_dir()
        downloaded = Path(client.download_artifacts(run.info.run_id, "release"))
        for name in RELEASE_FILES:
            assert (kwargs["out_dir"] / name).read_bytes() == baseline[name]
            assert (downloaded / name).read_bytes() == baseline[name]
        metadata = json.loads(baseline["metadata.json"])
        assert metadata["artifact_integrity"]["expected_sha256"] == training._artifact_manifest(
            kwargs["out_dir"]
        )
        assert metadata["split_policy"]["reported_metrics"] == "holdout_test"
        for model in ("rf", "xgb"):
            calibrated = joblib.load(kwargs["out_dir"] / f"{model}_calibrated.joblib")
            assert calibrated.predict_proba(frame.drop(columns="Class")).shape == (200, 2)
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_disabled_tracking_does_not_import_mlflow(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    work = MagicMock()
    monkeypatch.setattr(training, "_train", work)
    training.train(**_train_kwargs(tmp_path), tracking_enabled=False)
    assert work.call_args.kwargs == {"tracking": None, "register_models": False}


@pytest.mark.parametrize(
    "explicit,environment,expected",
    [
        (None, None, "sqlite:///mlflow.db"),
        (None, "sqlite:///env.db", "sqlite:///env.db"),
        ("sqlite:///flag.db", "sqlite:///env.db", "sqlite:///flag.db"),
    ],
)
def test_tracking_configuration(tmp_path, monkeypatch, explicit, environment, expected):
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    if environment:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", environment)
    else:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(training, "_train", MagicMock())
    training.train(
        **_train_kwargs(tmp_path), tracking_uri=explicit, experiment_name="test-training"
    )
    fake.set_tracking_uri.assert_called_once_with(expected)
    fake.set_experiment.assert_called_once_with("test-training")
    fake.start_run.assert_called_once_with(
        experiment_id=fake.set_experiment.return_value.experiment_id
    )


def test_training_failure_marks_redirected_run_failed(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'failed.db').as_posix()}"
    original_uri = mlflow.get_tracking_uri()
    try:
        with pytest.raises(FileNotFoundError):
            training.train(**_train_kwargs(tmp_path), tracking_uri=uri, experiment_name="failure")
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("failure")
        (run,) = client.search_runs([experiment.experiment_id])
        assert run.info.status == "FAILED"
        assert client.list_artifacts(run.info.run_id) == []
        assert mlflow.active_run() is None
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_logging_failure_preserves_release_and_skips_unavailable_auc(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kwargs = _train_kwargs(tmp_path)
    pd.DataFrame({"V1": [0, 1] * 100, "Class": [0, 1] * 100}).to_csv(
        kwargs["data_path"], index=False
    )
    monkeypatch.setattr(training, "_safe_auc", lambda *_: None)
    original_uri = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'logging-failure.db').as_posix()}"
    try:
        with (
            patch.object(mlflow, "log_artifact", side_effect=OSError("artifact store unavailable")),
            pytest.raises(OSError, match="artifact store unavailable"),
        ):
            training.train(**kwargs, tracking_uri=uri, experiment_name="logging-failure")
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name("logging-failure")
        (run,) = client.search_runs([experiment.experiment_id])
        assert run.info.status == "FAILED"
        assert len(run.data.metrics) == 44
        assert not any(key.endswith("roc_auc") for key in run.data.metrics)
        assert {path.name for path in kwargs["out_dir"].iterdir()} == RELEASE_FILES
        metadata = json.loads((kwargs["out_dir"] / "metadata.json").read_text())
        assert metadata["artifact_integrity"]["expected_sha256"] == training._artifact_manifest(
            kwargs["out_dir"]
        )
        assert mlflow.active_run() is None
    finally:
        mlflow.set_tracking_uri(original_uri)


def test_cli_tracking_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--data",
            "input.csv",
            "--no-tracking",
            "--tracking-uri",
            "sqlite:///test.db",
            "--experiment-name",
            "isolated",
        ],
    )
    work = MagicMock()
    monkeypatch.setattr(training, "train", work)
    training.main()
    assert work.call_args.kwargs["tracking_enabled"] is False
    assert work.call_args.kwargs["tracking_uri"] == "sqlite:///test.db"
    assert work.call_args.kwargs["experiment_name"] == "isolated"
    assert work.call_args.kwargs["register_models"] is False
