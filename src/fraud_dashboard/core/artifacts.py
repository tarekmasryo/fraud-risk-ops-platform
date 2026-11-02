from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib

try:  # pragma: no cover
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover
    InconsistentVersionWarning = Warning

from fraud_dashboard.core.config import get_settings
from fraud_dashboard.core.errors import ArtifactContractError
from fraud_dashboard.core.thresholds import normalize_model_key


def project_root() -> Path:
    """Return the repository root.

    This file lives at: src/fraud_dashboard/core/artifacts.py
    So: core -> fraud_dashboard -> src -> repo_root
    """

    return Path(__file__).resolve().parents[3]


def artifacts_dir(root: Path | None = None) -> Path:
    if root is not None:
        return root / "artifacts"
    artifact_dir = Path(get_settings().model_artifact_dir)
    if artifact_dir.is_absolute():
        return artifact_dir
    return project_root() / artifact_dir


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(root: Path | None = None) -> dict[str, Any]:
    adir = artifacts_dir(root)
    meta_path = adir / "metadata.json"
    if not meta_path.exists():
        return {}
    return _read_json(meta_path)


def load_thresholds(root: Path | None = None) -> dict[str, Any]:
    adir = artifacts_dir(root)
    raw = _read_json(adir / "thresholds.json")

    # Normalize model keys to canonical (rf / xgb) while preserving original keys.
    dm = raw.get("default_model")
    if isinstance(dm, str):
        raw["default_model"] = normalize_model_key(dm)

    models_cfg = raw.get("models")
    if isinstance(models_cfg, dict):
        normalized: dict[str, Any] = {}
        for k, v in models_cfg.items():
            nk = normalize_model_key(k) or k
            normalized[nk] = v
        # keep normalized copy (and keep original keys inside raw["models"] too)
        raw["models"].update(normalized)

    # Also normalize flat mapping format (rare but cheap)
    for k, v in list(raw.items()):
        nk = normalize_model_key(k)
        if nk and nk != k and isinstance(v, (int, float)):
            raw.setdefault(nk, v)

    return raw


def validate_artifacts(root: Path | None = None) -> None:
    """Validate trusted release metadata before any deserialization.

    The manifest must be deployed from a trusted source together with the models.
    This detects corruption, not an attacker replacing both files and manifest.
    """
    try:
        adir = artifacts_dir(root)
        metadata = load_metadata(root)
        env = metadata.get("env", {})
        if not all(env.get(key) for key in ("python", "scikit_learn", "xgboost")):
            raise ValueError("Missing artifact runtime version requirements")
        mismatches = check_env_versions(metadata)
        if mismatches:
            raise ValueError("; ".join(mismatches))
        expected = metadata.get("artifact_integrity", {}).get("expected_sha256", {})
        required = {p.name for p in adir.glob("*.joblib")}
        required.update(
            {"rf_calibrated.joblib", "xgb_calibrated.joblib", "policy.json", "thresholds.json"}
        )
        if not isinstance(expected, dict) or not required.issubset(expected):
            raise ValueError("Missing checksum manifest entries for model or policy artifacts")
        for name, digest in expected.items():
            if Path(name).name != name:
                raise ValueError("Invalid artifact manifest filename")
            with (adir / name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ArtifactContractError(f"Artifact validation failed: {exc}") from exc


def load_models(root: Path | None = None) -> dict[str, Any]:
    """Load calibrated models if present.

    Expected filenames (kept to match the original UI):
      - rf_calibrated.joblib
      - xgb_calibrated.joblib

    Returns a dict keyed by canonical model ids:
      - "rf"
      - "xgb"
    """

    validate_artifacts(root)
    adir = artifacts_dir(root)
    models: dict[str, Any] = {}

    rf_path = adir / "rf_calibrated.joblib"
    if rf_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            models["rf"] = joblib.load(rf_path)

    xgb_path = adir / "xgb_calibrated.joblib"
    if xgb_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            # XGBoost emits a noisy warning when unpickling sklearn-wrapped models.
            # The artifact is trusted within this project; we keep the runtime logs clean.
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                # The emitted message is multi-line and may include a timestamp prefix.
                # ``filterwarnings`` uses ``re.match`` (not ``re.search``), so we use DOTALL.
                message=r"(?s).*If you are loading a serialized model.*",
            )
            models["xgb"] = joblib.load(xgb_path)

    if not models:
        raise FileNotFoundError(
            "No calibrated model artifacts found in artifacts/. "
            "Expected rf_calibrated.joblib and/or xgb_calibrated.joblib."
        )

    return models


def load_bundle(root: Path | None = None) -> dict[str, Any]:
    """Load all runtime artifacts as a dict.

    Tests and the API expect a mapping with stable keys.
    """

    models = load_models(root)
    metadata = load_metadata(root)
    thresholds = load_thresholds(root)
    try:
        from fraud_dashboard.core.policy import load_policy, policy_to_thresholds

        policy = load_policy(root)
        thresholds = {**thresholds, **policy_to_thresholds(policy)}
    except Exception:
        policy = {}
    schema = (metadata or {}).get("schema", {})

    return {
        "available_models": sorted(models.keys()),
        "metadata": metadata,
        "schema": schema,
        "thresholds": thresholds,
        "policy": policy,
        "models": models,
    }


def check_env_versions(metadata: dict[str, Any]) -> list[str]:
    """Return a list of human-readable environment mismatch warnings."""

    warnings: list[str] = []
    env = (metadata or {}).get("env", {})

    expected_py = env.get("python")
    if expected_py:
        cur_py = ".".join(map(str, sys.version_info[:3]))
        expected_major_minor = ".".join(str(expected_py).split(".")[:2])
        cur_major_minor = ".".join(map(str, sys.version_info[:2]))
        if cur_major_minor != expected_major_minor:
            warnings.append(f"Python mismatch: expected {expected_py}, running {cur_py}.")

    expected_skl = env.get("scikit_learn")
    if expected_skl:
        try:
            import sklearn

            cur_skl = sklearn.__version__
            if cur_skl != expected_skl:
                warnings.append(
                    f"scikit-learn mismatch: expected {expected_skl}, running {cur_skl}."
                )
        except Exception as exc:  # pragma: no cover
            warnings.append(f"Could not import scikit-learn to validate version: {exc}")

    expected_xgb = env.get("xgboost")
    if expected_xgb:
        try:
            import xgboost

            cur_xgb = xgboost.__version__
            if cur_xgb != expected_xgb:
                warnings.append(f"xgboost mismatch: expected {expected_xgb}, running {cur_xgb}.")
        except Exception as exc:  # pragma: no cover
            warnings.append(f"Could not import xgboost to validate version: {exc}")

    return warnings
