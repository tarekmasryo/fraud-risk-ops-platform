from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    timeout_s: float = 10.0
    api_key: str | None = None
    bearer_token: str | None = None


class ApiError(RuntimeError):
    """Raised when the FastAPI backend cannot be reached or returns an error."""


def configured_api_url() -> str:
    """Use only a server-controlled API origin; never accept a browser-selected target.

    Credentials must not be forwarded to URL userinfo, arbitrary paths, queries or
    fragments. Local HTTP and Docker service DNS remain supported for reference use.
    """
    raw = os.getenv("FRAUD_API_URL") or os.getenv("API_BASE_URL") or "http://127.0.0.1:8000"
    try:
        url = urlsplit(raw)
        valid = (
            bool(raw)
            and not any(ord(ch) < 33 or ord(ch) == 127 for ch in raw)
            and url.scheme in {"http", "https"}
            and bool(url.hostname)
            and url.username is None
            and url.password is None
            and url.path in {"", "/"}
            and not url.query
            and not url.fragment
        )
        _ = url.port  # Reject malformed port numbers.
    except ValueError as exc:
        raise ValueError("Invalid server API URL configuration") from exc
    if not valid:
        raise ValueError("Invalid server API URL configuration")
    return raw.rstrip("/")


def get_api_config() -> ApiConfig:
    """Resolve the FastAPI base URL and optional auth from environment variables.

    - Preferred URL: FRAUD_API_URL
    - Back-compat URL: API_BASE_URL
    - Optional protected-mode API key: FRAUD_API_KEY
    - Optional protected-mode bearer token: FRAUD_BEARER_TOKEN
    """

    return ApiConfig(
        base_url=configured_api_url(),
        api_key=os.getenv("FRAUD_API_KEY") or None,
        bearer_token=os.getenv("FRAUD_BEARER_TOKEN") or None,
    )


def _auth_headers(cfg: ApiConfig) -> dict[str, str]:
    headers: dict[str, str] = {}
    if cfg.api_key:
        headers["X-API-Key"] = cfg.api_key
    if cfg.bearer_token:
        token = cfg.bearer_token.removeprefix("Bearer ").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _client(cfg: ApiConfig) -> httpx.Client:
    return httpx.Client(base_url=cfg.base_url, timeout=cfg.timeout_s, headers=_auth_headers(cfg))


def _request_json(
    method: str, path: str, *, cfg: ApiConfig, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        with _client(cfg) as c:
            r = c.request(method, path, json=payload)
        r.raise_for_status()
        data = r.json()
        # Attach a lightweight client-side timing too (useful for UI debugging).
        data.setdefault("client_latency_ms", (time.perf_counter() - t0) * 1000.0)
        return data
    except httpx.RequestError as e:
        raise ApiError(f"Cannot reach API at {cfg.base_url}: {e}") from e
    except httpx.HTTPStatusError as e:
        body = e.response.text
        raise ApiError(f"API error {e.response.status_code} on {path}: {body}") from e
    except ValueError as e:
        raise ApiError(f"API returned non-JSON response on {path}") from e


def ping(cfg: ApiConfig) -> dict[str, Any]:
    """GET /health"""

    return _request_json("GET", "/health", cfg=cfg)


def ping_ok(cfg: ApiConfig) -> bool:
    """Return True if GET /health succeeds."""
    try:
        data = ping(cfg)
        return bool(data.get("status") == "ok")
    except ApiError:
        return False


def fetch_metadata(cfg: ApiConfig) -> dict[str, Any]:
    """GET /metadata"""

    return _request_json("GET", "/metadata", cfg=cfg)


def predict_one(
    *,
    cfg: ApiConfig,
    model: str,
    record: dict[str, float],
    threshold: float | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """POST /v1/predictions."""

    payload: dict[str, Any] = {"model": model, "record": record}
    if policy is not None:
        payload["policy"] = policy
    if threshold is not None:
        payload["threshold"] = float(threshold)
    return _request_json("POST", "/v1/predictions", cfg=cfg, payload=payload)


def predict_batch(
    *,
    cfg: ApiConfig,
    model: str,
    records: list[dict[str, float]],
    threshold: float | None = None,
    policy: str | None = None,
) -> dict[str, Any]:
    """POST /v1/predictions/batch."""

    payload: dict[str, Any] = {"model": model, "records": records}
    if policy is not None:
        payload["policy"] = policy
    if threshold is not None:
        payload["threshold"] = float(threshold)
    return _request_json("POST", "/v1/predictions/batch", cfg=cfg, payload=payload)
