from __future__ import annotations

import logging
import time
import uuid
from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fraud_dashboard.api.schemas import (
    BatchJobCreateResponse,
    LoginRequest,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
)
from fraud_dashboard.core.artifact_contract import (
    artifact_checksums,
    check_artifact_contract,
)
from fraud_dashboard.core.artifacts import load_bundle, load_metadata, load_thresholds
from fraud_dashboard.core.config import get_settings, validate_settings
from fraud_dashboard.core.errors import ArtifactContractError, PredictionRuntimeError
from fraud_dashboard.core.logging import configure_logging
from fraud_dashboard.core.policy import (
    get_policy_threshold,
    load_policy,
    policy_to_thresholds,
)
from fraud_dashboard.core.security import (
    create_access_token,
    require_principal,
    require_role,
)
from fraud_dashboard.core.thresholds import normalize_model_key, pick_model_key
from fraud_dashboard.observability.metrics import (
    BATCH_JOBS_TOTAL,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    render_metrics,
)
from fraud_dashboard.platform.jobs import submit_background
from fraud_dashboard.platform.queue import enqueue_job
from fraud_dashboard.platform.store import get_store
from fraud_dashboard.services.batch_jobs import run_batch_job
from fraud_dashboard.services.metrics_summary import build_metrics_summary
from fraud_dashboard.services.reference_data import seed_reference_data
from fraud_dashboard.services.scoring import FraudScoringService

configure_logging()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    return load_bundle()


@lru_cache(maxsize=1)
def _policy() -> dict[str, Any]:
    return load_policy()


def _scoring_service() -> FraudScoringService:
    return FraudScoringService(bundle_loader=_bundle)


def _reference_bundle() -> dict[str, Any]:
    """Load metadata/policy references without unpickling model artifacts."""

    metadata = load_metadata()
    policy = _policy()
    try:
        thresholds = {**load_thresholds(), **policy_to_thresholds(policy)}
    except Exception:
        thresholds = policy_to_thresholds(policy)

    model_keys: set[str] = set()
    for raw_key in metadata.get("models") or {}:
        model_key = normalize_model_key(raw_key) or str(raw_key)
        model_keys.add(model_key)
    for policy_doc in (policy.get("policies") or {}).values():
        for raw_key in policy_doc.get("thresholds") or {}:
            model_key = normalize_model_key(raw_key) or str(raw_key)
            model_keys.add(model_key)

    return {
        "available_models": sorted(model_keys),
        "metadata": metadata,
        "schema": (metadata or {}).get("schema", {}),
        "thresholds": thresholds,
        "policy": policy,
    }


def _seed_reference_tables() -> bool:
    seed_reference_data(
        get_store(),
        bundle=_reference_bundle(),
        policy=_policy(),
        checksums=artifact_checksums(),
    )
    return True


def _resolve_policy(
    model: str | None, threshold: float | None, policy_name: str | None = None
) -> tuple[str, float, str]:
    return _scoring_service().resolve_policy(model, threshold, policy_name)


def _score_records(
    records: list[dict[str, Any]],
    *,
    model: str | None,
    threshold: float | None,
    policy: str | None,
    source: str,
) -> PredictBatchResponse:
    return _scoring_service().score_records(
        records, model=model, threshold=threshold, policy=policy, source=source
    )


def _run_job(job_id: str, req: PredictBatchRequest) -> None:
    run_batch_job(job_id, req, scoring_service=_scoring_service())


def create_app() -> FastAPI:
    validate_settings()
    app = FastAPI(
        title="Fraud Risk Ops API",
        version="0.1.0",
        description=(
            "Production-style reference API for fraud scoring, policy governance, "
            "audit logging, jobs, and monitoring."
        ),
    )
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allow_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("X-Request-ID", f"req_{uuid.uuid4().hex[:16]}")
        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        metric_path = getattr(route, "path", request.url.path)
        if get_settings().prometheus_enabled and HTTP_REQUESTS_TOTAL is not None:
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method, path=metric_path, status=str(response.status_code)
            ).inc()
            HTTP_REQUEST_DURATION.labels(method=request.method, path=metric_path).observe(
                latency_ms / 1000.0
            )
        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response

    @app.exception_handler(ArtifactContractError)
    async def artifact_error(_request: Request, exc: ArtifactContractError) -> JSONResponse:
        logger.error("artifact_validation_failed", extra={"error": str(exc)})
        return JSONResponse(
            status_code=503, content={"detail": "Model artifacts failed validation."}
        )

    @app.exception_handler(PredictionRuntimeError)
    async def prediction_error_handler(  # type: ignore[no-untyped-def]
        _request: Request, exc: PredictionRuntimeError
    ):
        return JSONResponse(content={"detail": str(exc)}, status_code=503)

    router = APIRouter(prefix="/v1")

    @app.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {
            "message": (
                "Use /docs. Main endpoints: /live, /ready, /v1/predictions, "
                "/v1/batch-jobs, /v1/metrics/summary, /metrics."
            )
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health")
    def health() -> dict[str, str]:
        # Backward-compatible lightweight health endpoint.
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        result = check_artifact_contract(run_sample_prediction=True)
        status_code = 200 if result.ok else 503
        if status_code != 200:
            raise HTTPException(status_code=status_code, detail=result.__dict__)
        return result.__dict__

    @app.get("/metadata")
    def metadata(
        _principal: Annotated[dict[str, Any], Depends(require_principal)],
    ) -> dict[str, Any]:
        _seed_reference_tables()
        b = _reference_bundle()
        models = {model_key: object() for model_key in b.get("available_models", [])}
        policy = _policy()
        thresholds = b.get("thresholds", {})
        tbm: dict[str, float] = {}
        for mk in sorted(models.keys()):
            try:
                tbm[mk] = float(get_policy_threshold(policy, model_key=mk))
            except Exception:
                continue
        return {
            "available_models": b.get("available_models", []),
            "default_model": normalize_model_key(policy.get("default_model"))
            or (pick_model_key(models, thresholds) if models else None),
            "policy": policy,
            "thresholds": thresholds,
            "thresholds_by_model": tbm,
            "schema": b.get("schema", {}),
            "env": (b.get("metadata") or {}).get("env", {}),
            "limits": {"max_batch_records": get_settings().max_batch_records},
            "artifact_checksums": artifact_checksums(),
        }

    @app.get("/metrics")
    def metrics() -> Response:
        if not get_settings().prometheus_enabled:
            raise HTTPException(status_code=404, detail="Prometheus metrics are disabled")
        return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")

    @router.post("/auth/login")
    def login(req: LoginRequest) -> dict[str, Any]:
        settings = get_settings()
        if req.username != settings.admin_username or req.password != settings.admin_password:
            get_store().audit(action="login_failed", resource_type="user", resource_id=req.username)
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token(req.username, role="admin")
        get_store().audit(action="login_success", resource_type="user", resource_id=req.username)
        return {"access_token": token, "token_type": "bearer", "role": "admin"}

    @router.get("/me")
    def me(principal: Annotated[dict[str, Any], Depends(require_principal)]) -> dict[str, Any]:
        return principal

    @router.post("/predictions", response_model=PredictResponse)
    def predict(
        req: PredictRequest, principal: Annotated[dict[str, Any], Depends(require_principal)]
    ) -> PredictResponse:
        out = _score_records(
            [req.record],
            model=req.model,
            threshold=req.threshold,
            policy=req.policy,
            source="api_single",
        )
        get_store().audit(
            action="prediction_api_single",
            actor_id=principal.get("sub"),
            resource_type="prediction_request",
            resource_id=out.request_id,
        )
        return out.results[0]

    @router.post("/predictions/batch", response_model=PredictBatchResponse)
    def predict_batch(
        req: PredictBatchRequest, principal: Annotated[dict[str, Any], Depends(require_principal)]
    ) -> PredictBatchResponse:
        out = _score_records(
            req.records,
            model=req.model,
            threshold=req.threshold,
            policy=req.policy,
            source="api_batch",
        )
        get_store().audit(
            action="prediction_api_batch",
            actor_id=principal.get("sub"),
            resource_type="prediction_request",
            resource_id=out.request_id,
            metadata={"records": len(req.records)},
        )
        return out

    @router.post("/batch-jobs", response_model=BatchJobCreateResponse)
    def create_batch_job(
        req: PredictBatchRequest, _principal: Annotated[dict[str, Any], Depends(require_principal)]
    ) -> BatchJobCreateResponse:
        model_key, th, _ = _resolve_policy(req.model, req.threshold, req.policy)
        job_id = get_store().create_job(
            model_key=model_key,
            threshold=th,
            total_records=len(req.records),
            request=req.model_dump(),
        )
        if get_settings().run_jobs_in_api:
            submit_background(_run_job, job_id, req)
        else:
            enqueue_job(job_id)
        if get_settings().prometheus_enabled and BATCH_JOBS_TOTAL is not None:
            BATCH_JOBS_TOTAL.labels(status="queued").inc()
        return BatchJobCreateResponse(job_id=job_id, status="queued")

    @router.get("/jobs/{job_id}")
    def get_job(
        job_id: str, _principal: Annotated[dict[str, Any], Depends(require_principal)]
    ) -> dict[str, Any]:
        job = get_store().get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @router.get("/audit-logs")
    def audit_logs(
        _principal: Annotated[dict[str, Any], Depends(require_role("admin", "analyst", "service"))],
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> list[dict[str, Any]]:
        return get_store().latest_audit_logs(limit=limit, offset=offset)

    @router.get("/policies")
    def policies(
        _principal: Annotated[dict[str, Any], Depends(require_principal)],
    ) -> dict[str, Any]:
        _seed_reference_tables()
        return _policy()

    @router.get("/metrics/summary")
    def metrics_summary(
        _principal: Annotated[dict[str, Any], Depends(require_role("admin", "analyst", "service"))],
    ) -> dict[str, Any]:
        _seed_reference_tables()
        return build_metrics_summary(get_store())

    @router.get("/model-versions")
    def model_versions(
        _principal: Annotated[dict[str, Any], Depends(require_principal)],
    ) -> dict[str, Any]:
        _seed_reference_tables()
        b = _reference_bundle()
        metadata = b.get("metadata", {})
        models = metadata.get("models", {})
        return {
            "models": models,
            "records": get_store().list_model_versions(),
            "artifact_checksums": artifact_checksums(),
            "env": metadata.get("env", {}),
        }

    app.include_router(router)

    # Legacy endpoints kept for existing UI/users.
    @app.post("/predict", response_model=PredictResponse)
    def legacy_predict(
        req: PredictRequest,
        _principal: Annotated[dict[str, Any], Depends(require_principal)],
        model: str | None = Query(default=None),
        threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> PredictResponse:
        out = _score_records(
            [req.record],
            model=model if model is not None else req.model,
            threshold=threshold if threshold is not None else req.threshold,
            policy=req.policy,
            source="legacy_single",
        )
        return out.results[0]

    @app.post("/predict/batch", response_model=PredictBatchResponse)
    def legacy_predict_batch(
        req: PredictBatchRequest,
        _principal: Annotated[dict[str, Any], Depends(require_principal)],
        model: str | None = Query(default=None),
        threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    ) -> PredictBatchResponse:
        return _score_records(
            req.records,
            model=model if model is not None else req.model,
            threshold=threshold if threshold is not None else req.threshold,
            policy=req.policy,
            source="legacy_batch",
        )

    return app


app = create_app()
