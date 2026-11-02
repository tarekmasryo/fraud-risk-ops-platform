from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_dashboard.core.artifacts import (
    artifacts_dir,
    check_env_versions,
    load_metadata,
    load_models,
)
from fraud_dashboard.core.errors import ArtifactContractError
from fraud_dashboard.core.policy import get_policy_threshold, load_policy
from fraud_dashboard.core.predict import predict_proba


@dataclass(frozen=True)
class ContractCheckResult:
    ok: bool
    status: str
    warnings: list[str]
    errors: list[str]
    checks: dict[str, bool]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_checksums(root: Path | None = None) -> dict[str, str]:
    adir = artifacts_dir(root)
    checksums: dict[str, str] = {}
    for path in sorted(adir.glob("*.joblib")):
        checksums[path.name] = sha256_file(path)
    for name in ("metadata.json", "thresholds.json", "policy.json"):
        path = adir / name
        if path.exists():
            checksums[name] = sha256_file(path)
    return checksums


def _sample_df(features: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{feature: 0.0 for feature in features}])


def _expected_checksums(metadata: dict) -> dict[str, str]:
    integrity = metadata.get("artifact_integrity") if isinstance(metadata, dict) else None
    expected = (integrity or {}).get("expected_sha256") if isinstance(integrity, dict) else None
    if not isinstance(expected, dict):
        return {}
    return {str(name): str(value) for name, value in expected.items()}


def _validate_expected_checksums(
    *, metadata: dict, warnings: list[str], errors: list[str], checks: dict[str, bool]
) -> None:
    current = artifact_checksums()
    expected = _expected_checksums(metadata)
    checks["artifact_checksums_available"] = bool(current)
    checks["artifact_checksum_manifest_present"] = bool(expected)
    if not expected:
        errors.append(
            "No expected checksum manifest found in metadata; "
            "scoring is disabled until a trusted manifest is supplied."
        )
        return

    missing = sorted(set(expected) - set(current))
    mismatched = sorted(
        name
        for name, digest in expected.items()
        if current.get(name) is not None and current[name] != digest
    )
    checks["artifact_checksums_match_manifest"] = not missing and not mismatched
    if missing:
        errors.append(f"Missing artifacts from checksum manifest: {missing}")
    if mismatched:
        errors.append(f"Artifact checksum mismatch: {mismatched}")


def check_artifact_contract(*, run_sample_prediction: bool = True) -> ContractCheckResult:
    warnings: list[str] = []
    errors: list[str] = []
    checks: dict[str, bool] = {}

    try:
        metadata = load_metadata()
        policy = load_policy()
    except (OSError, ValueError, TypeError) as exc:
        return ContractCheckResult(
            ok=False,
            status="not_ready",
            warnings=[],
            errors=[f"Cannot read artifact metadata or policy: {exc}"],
            checks={"sample_prediction": False},
        )

    checks["metadata_present"] = bool(metadata)
    if not metadata:
        errors.append("artifacts/metadata.json is missing or empty.")

    schema = metadata.get("schema") if isinstance(metadata, dict) else None
    features = (schema or {}).get("features") if isinstance(schema, dict) else None
    checks["schema_features_present"] = isinstance(features, list) and len(features) > 0
    if not checks["schema_features_present"]:
        errors.append("metadata.schema.features must contain the inference feature list.")

    checks["policy_present"] = bool(policy.get("policies"))
    if not checks["policy_present"]:
        errors.append("artifacts/policy.json must define at least one policy.")

    _validate_expected_checksums(metadata=metadata, warnings=warnings, errors=errors, checks=checks)

    warnings.extend(check_env_versions(metadata))
    runtime_matches = not warnings
    checks["runtime_matches_metadata"] = runtime_matches
    strict_runtime_failed = bool(warnings)
    if strict_runtime_failed:
        errors.append(
            "Artifact runtime mismatch. Recreate the environment from requirements.txt, "
            "or deploy artifacts compatible with this runtime."
        )

    adir = artifacts_dir()
    for name in ("rf_calibrated.joblib", "xgb_calibrated.joblib"):
        exists = (adir / name).exists()
        checks[f"{name}_present"] = exists
        if not exists:
            errors.append(f"Missing model artifact: {name}")

    if errors:
        checks["sample_prediction"] = False
    elif run_sample_prediction and checks.get("schema_features_present"):
        try:
            models = load_models()
            for model_key, model in models.items():
                _ = get_policy_threshold(policy, model_key=model_key)
                proba = predict_proba(
                    model, _sample_df(list(features)), allow_compatibility_fallback=False
                )
                if len(proba) != 1:
                    raise ArtifactContractError(
                        f"{model_key} returned invalid sample output length."
                    )
            checks["sample_prediction"] = True
        except Exception as exc:  # intentionally surface as readiness failure
            checks["sample_prediction"] = False
            errors.append(f"Sample prediction failed: {type(exc).__name__}: {exc}")
    else:
        checks["sample_prediction"] = False

    ok = not errors
    status = "ready" if ok else "not_ready"
    return ContractCheckResult(
        ok=ok, status=status, warnings=warnings, errors=errors, checks=checks
    )


def assert_artifact_contract() -> None:
    result = check_artifact_contract(run_sample_prediction=True)
    if not result.ok:
        raise ArtifactContractError("; ".join(result.errors))
