from __future__ import annotations

import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import HTTPException

from fraud_dashboard.api.schemas import PredictBatchResponse, PredictResponse
from fraud_dashboard.core.artifact_contract import artifact_checksums
from fraud_dashboard.core.artifacts import artifacts_dir, load_bundle
from fraud_dashboard.core.config import get_settings
from fraud_dashboard.core.decision import (
    decision_from_score,
    reason_codes_for_record,
    risk_band,
    stable_input_hash,
)
from fraud_dashboard.core.errors import ArtifactContractError, PredictionRuntimeError
from fraud_dashboard.core.policy import (
    get_default_policy_name,
    get_policy_threshold,
)
from fraud_dashboard.core.predict import apply_threshold, predict_proba
from fraud_dashboard.core.thresholds import (
    get_model_threshold,
    normalize_model_key,
    pick_model_key,
)
from fraud_dashboard.data.validation import (
    coerce_numeric,
    validate_columns,
    validate_finite,
    validate_no_nan,
)
from fraud_dashboard.observability.metrics import (
    HIGH_RISK_TOTAL,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY,
    PREDICTION_REQUESTS_TOTAL,
    THRESHOLD_POLICY_USAGE,
)
from fraud_dashboard.platform.store import OpsStore, get_store
from fraud_dashboard.services.reference_data import seed_reference_data

BundleLoader = Callable[[], dict[str, Any]]
PolicyLoader = Callable[[], dict[str, Any]]
StoreFactory = Callable[[], OpsStore]


@lru_cache(maxsize=1)
def _validated_bundle(artifact_path: str) -> dict[str, Any]:
    # One immutable release per process; restart API/worker after artifact deployment.
    return load_bundle()


def _default_bundle() -> dict[str, Any]:
    return _validated_bundle(str(artifacts_dir().resolve()))


class FraudScoringService:
    """Application service for fraud scoring and decision persistence.

    The API and worker both depend on this service so batch-job execution does not
    import private FastAPI module functions. The service owns model/policy
    resolution, feature validation, scoring, decision shaping, persistence, and
    prediction-level metrics.
    """

    def __init__(
        self,
        *,
        bundle_loader: BundleLoader = _default_bundle,
        policy_loader: PolicyLoader | None = None,
        store_factory: StoreFactory = get_store,
    ) -> None:
        self._bundle_loader = bundle_loader
        self._policy_loader = policy_loader
        self._store_factory = store_factory

    def bundle(self) -> dict[str, Any]:
        return self._bundle_loader()

    def policy(self) -> dict[str, Any]:
        return self._policy_loader() if self._policy_loader else self.bundle()["policy"]

    def store(self) -> OpsStore:
        return self._store_factory()

    def seed_reference_tables(self) -> bool:
        seed_reference_data(
            self.store(),
            bundle=self.bundle(),
            policy=self.policy(),
            checksums=artifact_checksums(),
        )
        return True

    def required_features(self) -> list[str]:
        schema = self.bundle().get("schema", {})
        feats = schema.get("features", [])
        if not feats:
            raise ArtifactContractError("metadata.json missing schema.features")
        return list(feats)

    def make_dataframe(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        try:
            validate_columns(df, self.required_features())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        df = df[self.required_features()]
        df = coerce_numeric(df, self.required_features())
        try:
            validate_no_nan(df, self.required_features())
            validate_finite(df, self.required_features())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return df

    def resolve_policy(
        self, model: str | None, threshold: float | None, policy_name: str | None = None
    ) -> tuple[str, float, str]:
        """Resolve model and operating threshold for a request.

        Explicit request fields fail loudly when invalid. Silent fallback is
        useful for legacy artifact migration, but dangerous when a caller asks
        for a named policy and receives a different policy without knowing it.
        """

        bundle = self.bundle()
        models = bundle["models"]
        thresholds = bundle["thresholds"]
        policy_doc = self.policy()

        requested_model = normalize_model_key(model)
        model_key = (
            requested_model
            or normalize_model_key(policy_doc.get("default_model"))
            or pick_model_key(models, thresholds)
        )

        if model_key not in models:
            raise HTTPException(
                status_code=400,
                detail=(f"Unknown model '{model_key}'. Available: {sorted(models.keys())}"),
            )

        if threshold is not None:
            resolved_policy = "manual_override"
            resolved_threshold = float(threshold)
        else:
            available_policies = sorted((policy_doc.get("policies") or {}).keys())
            if policy_name is not None and policy_name not in available_policies:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Unknown policy '{policy_name}'. Available: {available_policies}"),
                )

            resolved_policy = policy_name or get_default_policy_name(policy_doc)
            try:
                resolved_threshold = float(
                    get_policy_threshold(
                        policy_doc, model_key=model_key, policy_name=resolved_policy
                    )
                )
            except KeyError as exc:
                if policy_doc.get("policies"):
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                resolved_threshold = float(get_model_threshold(thresholds, model_key))
                resolved_policy = "legacy_thresholds"

        if not (0.0 <= resolved_threshold <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid threshold for '{model_key}': {resolved_threshold}",
            )
        return model_key, resolved_threshold, resolved_policy

    def score_records(
        self,
        records: list[dict[str, Any]],
        *,
        model: str | None,
        threshold: float | None,
        policy: str | None,
        source: str,
    ) -> PredictBatchResponse:
        t0 = time.perf_counter()
        model_key, th, resolved_policy = self.resolve_policy(model, threshold, policy)
        df = self.make_dataframe(records)
        try:
            probs = predict_proba(self.bundle()["models"][model_key], df)
            preds = apply_threshold(probs, th)
        except Exception as exc:
            if get_settings().prometheus_enabled and PREDICTION_ERRORS_TOTAL is not None:
                PREDICTION_ERRORS_TOTAL.labels(model=model_key, error_type=type(exc).__name__).inc()
            raise PredictionRuntimeError(f"Model scoring failed for '{model_key}': {exc}") from exc

        dt_ms = int((time.perf_counter() - t0) * 1000)
        policy_doc = self.policy()
        policy_version = str(policy_doc.get("policy_version", "policy"))
        record_details = []
        for record, score, label in zip(records, probs.tolist(), preds.tolist(), strict=False):
            decision, review_required = decision_from_score(float(score), th)
            record_details.append(
                {
                    "risk_score": float(score),
                    "label": int(label),
                    "decision": decision,
                    "review_required": review_required,
                    "risk_band": risk_band(float(score)),
                    "reason_codes": reason_codes_for_record(
                        record, score=float(score), threshold=th
                    ),
                    "input_hash": stable_input_hash(record),
                }
            )

        request_id = self.store().record_prediction_request(
            model_key=model_key,
            threshold=th,
            probabilities=[float(p) for p in probs.tolist()],
            labels=[int(y) for y in preds.tolist()],
            latency_ms=dt_ms,
            source=source,
            policy_name=resolved_policy,
            policy_version=policy_version,
            decisions=[str(item["decision"]) for item in record_details],
            reason_codes=[list(item["reason_codes"]) for item in record_details],
            input_hashes=[str(item["input_hash"]) for item in record_details],
        )

        if get_settings().prometheus_enabled and PREDICTION_REQUESTS_TOTAL is not None:
            PREDICTION_REQUESTS_TOTAL.labels(model=model_key, source=source).inc()
            PREDICTION_LATENCY.labels(model=model_key, source=source).observe(dt_ms / 1000.0)
            THRESHOLD_POLICY_USAGE.labels(model=model_key, policy=resolved_policy).inc()
            high_risk_count = int(sum(int(y) for y in preds.tolist()))
            if high_risk_count:
                HIGH_RISK_TOTAL.labels(model=model_key).inc(high_risk_count)

        results = [
            PredictResponse(
                model=model_key,
                threshold=th,
                proba_fraud=float(item["risk_score"]),
                risk_score=float(item["risk_score"]),
                label=int(item["label"]),
                decision=str(item["decision"]),
                review_required=bool(item["review_required"]),
                risk_band=str(item["risk_band"]),
                reason_codes=list(item["reason_codes"]),
                policy=resolved_policy,
                policy_version=policy_version,
                input_hash=str(item["input_hash"]),
                latency_ms=dt_ms,
                request_id=request_id,
            )
            for item in record_details
        ]
        return PredictBatchResponse(
            model=model_key,
            threshold=th,
            results=results,
            latency_ms=dt_ms,
            request_id=request_id,
        )
