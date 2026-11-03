from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - keeps non-UI unit tests lightweight

    class _StreamlitCacheFallback:
        @staticmethod
        def cache_resource(*_args, **_kwargs):
            def decorator(fn):
                return fn

            return decorator

    st = _StreamlitCacheFallback()

from fraud_dashboard.core.artifacts import load_bundle
from fraud_dashboard.core.config import get_settings
from fraud_dashboard.core.predict import predict_proba
from fraud_dashboard.ui.api_client import (
    ApiConfig,
    BackendUnavailableError,
    fetch_metadata,
    get_api_config,
    ping_ok,
    predict_batch,
)


@dataclass(frozen=True)
class BackendInfo:
    kind: str  # "api" or "local"
    details: str


class Predictor(Protocol):
    def info(self) -> BackendInfo: ...

    def available_models(self) -> list[str]: ...

    def thresholds(self) -> dict[str, Any]: ...

    def schema_features(self) -> list[str]: ...

    def predict_proba_batch(
        self,
        df_features: pd.DataFrame,
        *,
        model_key: str,
        threshold: float | None = None,
        policy: str | None = None,
    ) -> tuple[np.ndarray, float]: ...


@st.cache_resource(show_spinner=False)
def _local_bundle() -> dict[str, Any]:
    return load_bundle()


class LocalPredictor:
    last_results: list[dict[str, Any]] | None = None

    def __init__(self) -> None:
        self._bundle = _local_bundle()

    def info(self) -> BackendInfo:
        return BackendInfo(kind="local", details="Local artifacts (no API required)")

    def available_models(self) -> list[str]:
        return list(self._bundle.get("available_models", []))

    def thresholds(self) -> dict[str, Any]:
        return dict(self._bundle.get("thresholds", {}))

    def schema_features(self) -> list[str]:
        schema = self._bundle.get("schema", {})
        feats = schema.get("features") or []
        return list(feats)

    def predict_proba_batch(
        self,
        df_features: pd.DataFrame,
        *,
        model_key: str,
        threshold: float | None = None,
        policy: str | None = None,
    ) -> tuple[np.ndarray, float]:
        _ = threshold, policy
        t0 = time.perf_counter()
        model = self._bundle["models"][model_key]
        probs = predict_proba(model, df_features)
        dt = time.perf_counter() - t0
        return np.asarray(probs, dtype=float), float(dt)


class ApiPredictor:
    def __init__(self, cfg: ApiConfig) -> None:
        self.cfg = cfg
        self._meta: dict[str, Any] | None = None
        self.last_results: list[dict[str, Any]] | None = None

    def _metadata(self) -> dict[str, Any]:
        if self._meta is None:
            self._meta = fetch_metadata(self.cfg)
        return self._meta

    def info(self) -> BackendInfo:
        return BackendInfo(kind="api", details=f"FastAPI @ {self.cfg.base_url}")

    def available_models(self) -> list[str]:
        return list(self._metadata().get("available_models", []))

    def thresholds(self) -> dict[str, Any]:
        return dict(self._metadata().get("thresholds", {}))

    def schema_features(self) -> list[str]:
        schema = self._metadata().get("schema", {})
        feats = schema.get("features") or []
        return list(feats)

    def _max_batch_records(self) -> int:
        raw_limit = (self._metadata().get("limits") or {}).get("max_batch_records", 1000)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 1000
        return max(1, limit)

    def predict_proba_batch(
        self,
        df_features: pd.DataFrame,
        *,
        model_key: str,
        threshold: float | None = None,
        policy: str | None = None,
    ) -> tuple[np.ndarray, float]:
        self.last_results = None
        feats = list(df_features.columns)
        records = df_features[feats].to_dict(orient="records")
        chunk_size = self._max_batch_records()
        t0 = time.perf_counter()
        probs: list[float] = []
        results: list[dict[str, Any]] = []
        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            out = predict_batch(
                cfg=self.cfg,
                model=model_key,
                records=chunk,
                threshold=threshold if policy is None else None,
                policy=policy,
            )
            chunk_results = out.get("results", [])
            probs.extend(float(r["proba_fraud"]) for r in chunk_results)
            results.extend(dict(row) for row in chunk_results)
        dt = time.perf_counter() - t0
        if len(probs) != len(records):
            raise RuntimeError(
                f"API returned {len(probs)} predictions for {len(records)} input records."
            )
        self.last_results = results
        return np.asarray(probs, dtype=float), float(dt)


def resolve_predictor(
    mode: str,
    api_url: str,
    *,
    api_key: str | None = None,
    bearer_token: str | None = None,
) -> tuple[Predictor, str]:
    """Select predictor given mode.

    mode: "Auto", "API", "Local"
    returns: (predictor, status_message)
    """

    api_cfg = ApiConfig(
        base_url=api_url.rstrip("/"),
        timeout_s=15.0,
        api_key=api_key or None,
        bearer_token=bearer_token or None,
    )

    if mode == "Local":
        return LocalPredictor(), "Using local artifacts."
    if mode == "API":
        if not ping_ok(api_cfg):
            if get_settings().allow_local_fallback:
                return (
                    LocalPredictor(),
                    "API not reachable → fell back to local artifacts (local fallback enabled).",
                )
            raise BackendUnavailableError(
                "API mode selected but backend is not reachable and ALLOW_LOCAL_FALLBACK=false."
            )
        return ApiPredictor(api_cfg), "Using FastAPI backend."

    # Auto
    if ping_ok(api_cfg):
        return ApiPredictor(api_cfg), "Auto: using FastAPI backend."
    if get_settings().allow_local_fallback:
        return (
            LocalPredictor(),
            "Auto: API not reachable → using local artifacts (local fallback enabled).",
        )
    raise BackendUnavailableError("Auto mode cannot reach API and ALLOW_LOCAL_FALLBACK=false.")


def render_backend_unavailable() -> None:
    """End this run deliberately; never present an earlier prediction as current."""
    st.session_state.pop("last_run", None)
    st.error("Prediction API unavailable. No current prediction results are available.")
    st.info(
        "Restore the configured API service, then select Retry API connection. "
        "No local fallback has been activated by this error."
    )
    if st.button("Retry API connection", key="retry_prediction_api"):
        st.rerun()
    st.stop()


def ui_backend_selector() -> tuple[Predictor, dict[str, Any]]:
    st.sidebar.header("Backend")
    mode = st.sidebar.radio("Inference mode", ["Auto", "API", "Local"], horizontal=True)

    # Browser visitors cannot redirect authenticated server-side API calls.
    server_config = get_api_config()
    api_url = server_config.base_url
    st.sidebar.caption("API endpoint is configured on the server.")

    with st.sidebar.expander("Protected API auth", expanded=False):
        api_key_input = st.text_input(
            "API key",
            value="",
            type="password",
            key="manual_api_key_v2",
            help="Optional session credential. Server credentials are never displayed.",
        )
        bearer_input = st.text_input(
            "Bearer token",
            value="",
            type="password",
            key="manual_bearer_token_v2",
            help="Optional JWT for a trusted local review session.",
        )

    api_key_input = api_key_input.strip()
    bearer_input = bearer_input.strip()
    manual_override = bool(api_key_input or bearer_input)
    # Do not mix per-session overrides with other server-provided credentials.
    if manual_override:
        effective_api_key = api_key_input or None
        effective_bearer = bearer_input or None
    else:
        effective_api_key = server_config.api_key
        effective_bearer = server_config.bearer_token

    predictor, msg = resolve_predictor(
        mode,
        api_url,
        api_key=effective_api_key,
        bearer_token=effective_bearer,
    )

    if predictor.info().kind == "api":
        st.sidebar.success("API reachable")
    else:
        st.sidebar.info("Local mode")

    st.sidebar.caption(msg)
    meta = {
        "mode": mode,
        "api_url": api_url,
        "auth_configured": bool(effective_api_key or effective_bearer),
    }
    return predictor, meta
