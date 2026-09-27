from __future__ import annotations

import json
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from scripts import model_registry as registry
from scripts import train as training


@pytest.fixture
def registered_models(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    tracking_uri = mlflow.get_tracking_uri()
    registry_uri = mlflow.get_registry_uri()
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(160, 2)), columns=["V1", "Amount"])
    frame["Class"] = (frame.V1 + frame.Amount > 0).astype(int)
    data = tmp_path / "input.csv"
    frame.to_csv(data, index=False)
    features = tmp_path / "features.csv"
    frame.drop(columns="Class").iloc[:3].to_csv(features, index=False)
    # An unrelated registry setting must not divert local registration.
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", f"sqlite:///{tmp_path.as_posix()}/other.db")
    try:
        for seed in (42, 43):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "train.py",
                    "--data",
                    str(data),
                    "--out",
                    str(tmp_path / f"release-{seed}"),
                    "--seed",
                    str(seed),
                    "--register-models",
                ],
            )
            with monkeypatch.context() as context:
                if seed == 42:
                    context.setattr(training, "_safe_auc", lambda *_: None)
                training.main()
        mlflow.set_registry_uri(uri)
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        yield client, tmp_path, features
    finally:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)


def test_cli_versions_tags_alias_promotion_and_load(registered_models, capsys):
    client, root, features = registered_models
    for family in ("rf", "xgb"):
        name = f"fraud-risk-{family}"
        versions = registry.list_versions(client, name)
        assert [item["version"] for item in versions] == ["1", "2"]
        assert client.get_registered_model(name).aliases == {}
        for version in versions:
            run = client.get_run(version["run_id"])
            assert run.info.status == "FINISHED"
            tags = version["tags"]
            assert tags["model_family"] == family
            assert tags["run_id"] == run.info.run_id
            assert tags["dataset_sha256"] == training._sha256_file(root / "input.csv")
            for tag, metric in (
                ("holdout_f1", "f1"),
                ("holdout_roc_auc", "roc_auc"),
                ("holdout_average_cost", "average_cost"),
                ("threshold", "threshold"),
            ):
                metric_key = f"{family}.holdout_test.{metric}"
                if metric_key in run.data.metrics:
                    assert float(tags[tag]) == run.data.metrics[metric_key]
                else:
                    assert tag == "holdout_roc_auc"
                    assert tag not in tags
            info = mlflow.models.get_model_info(f"models:/{name}/{version['version']}")
            assert {"sklearn", "python_function"} <= info.flavors.keys()
            assert info.signature is not None
            assert info.saved_input_example_info is None

    def command(*args):
        capsys.readouterr()
        registry.main(list(args))
        return json.loads(capsys.readouterr().out)

    assert [item["version"] for item in command("list")] == ["1", "2"]
    assert command("aliases") == {}
    # Promotion cannot invent a candidate or select the best metric.
    with pytest.raises(SystemExit) as exc:
        registry.main(["promote"])
    assert exc.value.code == 1
    assert client.get_registered_model("fraud-risk-rf").aliases == {}

    for version, seed in (("1", 42), ("2", 43), ("1", 42)):
        previous_champion = command("aliases").get("champion")
        assert command("set-candidate", "--version", version)["version"] == version
        assert command("aliases").get("champion") == previous_champion
        if previous_champion is None:
            with pytest.raises(MlflowException):
                client.get_model_version_by_alias("fraud-risk-rf", "champion")
        assert command("promote")["version"] == version
        assert command("champion")["version"] == version
        assert command("aliases") == {"candidate": version, "champion": version}
        smoke = command("smoke", "--input-csv", str(features))
        assert smoke["version"] == version
        assert smoke["model_uri"] == "models:/fraud-risk-rf@champion"
        native = joblib.load(root / f"release-{seed}" / "rf_calibrated.joblib")
        np.testing.assert_allclose(
            smoke["probabilities"], native.predict_proba(pd.read_csv(features))
        )

    # XGB's calibrated sklearn wrapper is loadable too.
    command("--name", "fraud-risk-xgb", "set-candidate", "--version", "2")
    command("--name", "fraud-risk-xgb", "promote")
    smoke = command("--name", "fraud-risk-xgb", "smoke", "--input-csv", str(features))
    native = joblib.load(root / "release-43" / "xgb_calibrated.joblib")
    np.testing.assert_allclose(smoke["probabilities"], native.predict_proba(pd.read_csv(features)))
    with pytest.raises(SystemExit):
        registry.main(["set-candidate", "--version", "999"])
    assert command("aliases") == {"candidate": "1", "champion": "1"}
    pd.DataFrame({"wrong_feature": [1]}).to_csv(root / "bad.csv", index=False)
    with pytest.raises(SystemExit):
        registry.main(["smoke", "--input-csv", str(root / "bad.csv")])


def test_registration_requires_tracking(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--data", "input.csv", "--no-tracking", "--register-models"]
    )
    with pytest.raises(SystemExit) as exc:
        training.parse_args()
    assert exc.value.code == 2
    with pytest.raises(ValueError, match="requires tracking"):
        training.train(
            tmp_path / "missing.csv",
            tmp_path / "out",
            "Class",
            42,
            0.2,
            0.2,
            5,
            500,
            tracking_enabled=False,
            register_models=True,
        )
    assert not (tmp_path / "out").exists()
