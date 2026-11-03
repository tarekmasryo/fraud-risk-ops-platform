from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np

from fraud_dashboard.core import artifacts
from fraud_dashboard.core.artifacts import (
    check_env_versions,
    load_bundle,
    load_metadata,
    load_models,
    project_root,
)


class _TinyModel:
    def predict_proba(self, X):
        scores = np.full(len(X), 0.2, dtype=float)
        return np.column_stack([1.0 - scores, scores])


def test_project_root_exists() -> None:
    root = project_root()
    assert root.exists()
    assert (root / "artifacts").exists()


def test_load_metadata_has_schema() -> None:
    metadata = load_metadata()
    assert metadata["artifact_version"] == "fraud-risk-ops-v0.1.0-sklearn1.5.2"
    assert "features" in metadata["schema"]
    assert len(metadata["schema"]["features"]) > 10


def test_load_bundle_has_expected_keys(monkeypatch) -> None:
    monkeypatch.setattr(artifacts, "load_models", lambda root=None: {"rf": _TinyModel()})
    bundle = load_bundle()
    assert "available_models" in bundle
    assert "metadata" in bundle
    assert "schema" in bundle
    assert "thresholds" in bundle
    assert "models" in bundle
    assert set(bundle["models"].keys()).issuperset({"rf"})


def test_load_models_from_temp_artifacts(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    joblib.dump(_TinyModel(), artifact_dir / "rf_calibrated.joblib")
    joblib.dump(_TinyModel(), artifact_dir / "xgb_calibrated.joblib")
    for name in ("policy.json", "thresholds.json"):
        (artifact_dir / name).write_text("{}")
    metadata = load_metadata()
    metadata["artifact_integrity"]["expected_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact_dir.iterdir()
    }
    (artifact_dir / "metadata.json").write_text(json.dumps(metadata))

    models = load_models(root=tmp_path)
    assert "rf" in models


def test_check_env_versions_mismatch_warnings() -> None:
    meta = {"env": {"python": "0.0.0", "scikit_learn": "0.0.0", "xgboost": "0.0.0"}}
    warnings = check_env_versions(meta)
    assert len(warnings) >= 1


def test_metadata_json_is_valid() -> None:
    raw = (project_root() / "artifacts" / "metadata.json").read_text(encoding="utf-8")
    assert json.loads(raw)["release"]["version"] == "0.1.0"
