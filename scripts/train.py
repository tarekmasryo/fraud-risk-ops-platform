from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None  # type: ignore


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _safe_auc(y_true: np.ndarray, proba: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _eval_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    *,
    cost_fp: float,
    cost_fn: float,
) -> dict[str, Any]:
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = max(len(y_true), 1)
    total_cost = float(fp * cost_fp + fn * cost_fn)
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(y_true, proba),
        "review_rate": float((tp + fp) / total),
        "total_cost": total_cost,
        "average_cost": float(total_cost / total),
    }


def _pick_min_cost_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    cost_fp: float,
    cost_fn: float,
) -> float:
    """Pick an operating threshold on the calibration split only."""

    candidates = np.unique(np.concatenate([np.array([0.0, 1.0]), np.clip(proba, 0.0, 1.0)]))
    best_threshold = 0.5
    best_cost = float("inf")
    best_review_rate = float("inf")
    for threshold in candidates:
        metrics = _eval_at_threshold(
            y_true, proba, float(threshold), cost_fp=cost_fp, cost_fn=cost_fn
        )
        cost = float(metrics["total_cost"])
        review_rate = float(metrics["review_rate"])
        if (cost, review_rate, float(threshold)) < (best_cost, best_review_rate, best_threshold):
            best_threshold = float(threshold)
            best_cost = cost
            best_review_rate = review_rate
    return best_threshold


def _split_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    calibration_size: float,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    if calibration_size <= 0 or test_size <= 0:
        raise ValueError("calibration_size and test_size must both be positive.")
    temp_size = calibration_size + test_size
    if not 0 < temp_size < 1:
        raise ValueError("calibration_size + test_size must be between 0 and 1.")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=seed, stratify=y
    )
    relative_test_size = test_size / temp_size
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=seed,
        stratify=y_temp,
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_manifest(out_dir: Path) -> dict[str, str]:
    names = [
        "rf_calibrated.joblib",
        "rf_pipe.joblib",
        "xgb_calibrated.joblib",
        "xgb_pipe.joblib",
        "policy.json",
        "thresholds.json",
    ]
    return {name: _sha256_file(out_dir / name) for name in names if (out_dir / name).exists()}


def train(
    data_path: Path,
    out_dir: Path,
    label_col: str,
    seed: int,
    calibration_size: float,
    test_size: float,
    cost_fp: float,
    cost_fn: float,
    *,
    tracking_enabled: bool = True,
    tracking_uri: str | None = None,
    experiment_name: str = "fraud-risk-training",
    register_models: bool = False,
) -> None:
    """Train with optional, explicit tracking; keep release artifacts independent."""
    if register_models and not tracking_enabled:
        raise ValueError("--register-models requires tracking; remove --no-tracking.")
    with ExitStack() as stack:
        tracking = None
        if tracking_enabled:
            import mlflow

            mlflow.set_tracking_uri(
                tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///mlflow.db"
            )
            if register_models:
                registry_uri = mlflow.get_registry_uri()
                stack.callback(mlflow.set_registry_uri, registry_uri)
                mlflow.set_registry_uri(mlflow.get_tracking_uri())
            experiment = mlflow.set_experiment(experiment_name)
            stack.enter_context(mlflow.start_run(experiment_id=experiment.experiment_id))
            tracking = mlflow
        _train(
            data_path,
            out_dir,
            label_col,
            seed,
            calibration_size,
            test_size,
            cost_fp,
            cost_fn,
            tracking=tracking,
            register_models=register_models,
        )


def _train(
    data_path: Path,
    out_dir: Path,
    label_col: str,
    seed: int,
    calibration_size: float,
    test_size: float,
    cost_fp: float,
    cost_fn: float,
    *,
    tracking: Any = None,
    register_models: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {data_path}")

    df = df.dropna()
    y = df[label_col].astype(int).to_numpy()
    X = _ensure_numeric(df.drop(columns=[label_col]))

    X_train, X_cal, X_test, y_train, y_cal, y_test = _split_dataset(
        X,
        y,
        seed=seed,
        calibration_size=calibration_size,
        test_size=test_size,
    )
    if tracking is not None:
        tracking.log_params(
            {
                "seed": seed,
                "calibration_size": calibration_size,
                "test_size": test_size,
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
                "label_col": label_col,
                "dataset_name": data_path.name,
                "dataset_sha256": _sha256_file(data_path),
                "train_rows": len(X_train),
                "calibration_rows": len(X_cal),
                "test_rows": len(X_test),
                "feature_count": X.shape[1],
            }
        )

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    if tracking is not None:
        tracking.log_params({f"rf.{key}": value for key, value in rf.get_params().items()})
    rf.fit(X_train, y_train)
    rf_cal = CalibratedClassifierCV(rf, cv="prefit", method="sigmoid")
    rf_cal.fit(X_cal, y_cal)

    if xgb is None:
        raise RuntimeError(
            "xgboost is not installed. Install it (see requirements.txt) to train the XGBoost model."
        )

    neg = float((y_train == 0).sum())
    pos = float((y_train == 1).sum())
    scale_pos_weight = (neg / max(pos, 1.0)) if pos > 0 else 1.0

    xgb_clf = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    if tracking is not None:
        tracking.log_params({f"xgb.{key}": value for key, value in xgb_clf.get_params().items()})
    xgb_clf.fit(X_train, y_train)
    xgb_cal = CalibratedClassifierCV(xgb_clf, cv="prefit", method="sigmoid")
    xgb_cal.fit(X_cal, y_cal)

    rf_cal_proba = rf_cal.predict_proba(X_cal)[:, 1]
    xgb_cal_proba = xgb_cal.predict_proba(X_cal)[:, 1]
    rf_test_proba = rf_cal.predict_proba(X_test)[:, 1]
    xgb_test_proba = xgb_cal.predict_proba(X_test)[:, 1]

    rf_thr = _pick_min_cost_threshold(y_cal, rf_cal_proba, cost_fp=cost_fp, cost_fn=cost_fn)
    xgb_thr = _pick_min_cost_threshold(y_cal, xgb_cal_proba, cost_fp=cost_fp, cost_fn=cost_fn)

    rf_cal_eval = _eval_at_threshold(y_cal, rf_cal_proba, rf_thr, cost_fp=cost_fp, cost_fn=cost_fn)
    xgb_cal_eval = _eval_at_threshold(
        y_cal, xgb_cal_proba, xgb_thr, cost_fp=cost_fp, cost_fn=cost_fn
    )
    rf_holdout_eval = _eval_at_threshold(
        y_test, rf_test_proba, rf_thr, cost_fp=cost_fp, cost_fn=cost_fn
    )
    xgb_holdout_eval = _eval_at_threshold(
        y_test, xgb_test_proba, xgb_thr, cost_fp=cost_fp, cost_fn=cost_fn
    )

    thresholds = {
        "artifact_version": "fraud-risk-ops-v0.1.0",
        "default_model": "rf",
        "split_policy": "train/calibration/holdout_test",
        "selection_split": "calibration",
        "reporting_split": "holdout_test",
        "cost_settings": {"cost_fp": float(cost_fp), "cost_fn": float(cost_fn)},
        "models": {
            "rf": {
                "threshold": float(rf_thr),
                "threshold_selection": rf_cal_eval,
                "holdout_evaluation": rf_holdout_eval,
            },
            "xgb": {
                "threshold": float(xgb_thr),
                "threshold_selection": xgb_cal_eval,
                "holdout_evaluation": xgb_holdout_eval,
            },
        },
    }

    policy = {
        "policy_version": "fraud-risk-ops-v0.1.0",
        "release_stage": "public_reference",
        "default_model": "rf",
        "default_policy": "min_cost",
        "costs": {"false_positive": float(cost_fp), "false_negative": float(cost_fn)},
        "models": {
            "rf": {"display_name": "RandomForest (Calibrated)", "default_policy": "min_cost"},
            "xgb": {"display_name": "XGBoost (Calibrated)", "default_policy": "min_cost"},
        },
        "policies": {
            "strict": {
                "thresholds": {"rf": 0.90, "xgb": 0.90},
                "description": "Reduce false positives and analyst load.",
            },
            "balanced": {
                "thresholds": {"rf": 0.65, "xgb": 0.75},
                "description": "Default review-oriented operating point.",
            },
            "min_cost": {
                "thresholds": {"rf": float(rf_thr), "xgb": float(xgb_thr)},
                "description": "Cost-minimizing operating point selected on the calibration split.",
            },
            "lenient": {
                "thresholds": {"rf": 0.20, "xgb": 0.20},
                "description": "Increase fraud capture with higher review load.",
            },
        },
    }

    joblib.dump(rf, out_dir / "rf_pipe.joblib")
    joblib.dump(rf_cal, out_dir / "rf_calibrated.joblib")
    joblib.dump(xgb_clf, out_dir / "xgb_pipe.joblib")
    joblib.dump(xgb_cal, out_dir / "xgb_calibrated.joblib")
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (out_dir / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    metadata = {
        "artifact_version": "fraud-risk-ops-v0.1.0",
        "trained_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": __import__("sklearn").__version__,
            "xgboost": getattr(__import__("xgboost"), "__version__", None),
        },
        "release": {
            "project": "fraud-risk-ops-platform",
            "version": "0.1.0",
            "purpose": "ops_platform_reference_artifact",
            "runtime_policy": "fail_closed_on_artifact_runtime_mismatch",
        },
        "schema": {"label": label_col, "features": list(X.columns)},
        "thresholds_file": "policy.json",
        "split_policy": {
            "train_fraction": float(1.0 - calibration_size - test_size),
            "calibration_fraction": float(calibration_size),
            "holdout_test_fraction": float(test_size),
            "threshold_selection": "calibration",
            "reported_metrics": "holdout_test",
        },
        "models": {
            "random_forest": {"file": "rf_calibrated.joblib", "pipeline_file": "rf_pipe.joblib"},
            "xgboost": {"file": "xgb_calibrated.joblib", "pipeline_file": "xgb_pipe.joblib"},
        },
        "metrics": {
            "note": "Thresholds are selected on calibration data; reported metrics are computed on the holdout test split only.",
            "holdout_test": {
                "random_forest": rf_holdout_eval,
                "xgboost": xgb_holdout_eval,
            },
            "calibration_threshold_selection": {
                "random_forest": rf_cal_eval,
                "xgboost": xgb_cal_eval,
            },
        },
        "artifact_integrity": {
            "algorithm": "sha256",
            "expected_sha256": _artifact_manifest(out_dir),
            "note": "Readiness validates these expected checksums for model and policy artifacts. metadata.json stores the manifest and is reported separately.",
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if tracking is not None:
        for model, evaluations in thresholds["models"].items():
            for split, key in (
                ("calibration", "threshold_selection"),
                ("holdout_test", "holdout_evaluation"),
            ):
                tracking.log_metrics(
                    {
                        f"{model}.{split}.{name}": float(value)
                        for name, value in evaluations[key].items()
                        if value is not None
                    }
                )
        for name in (*metadata["artifact_integrity"]["expected_sha256"], "metadata.json"):
            tracking.log_artifact(str(out_dir / name), artifact_path="release")
        for family, estimator, evaluation in (
            ("rf", rf_cal, rf_holdout_eval),
            ("xgb", xgb_cal, xgb_holdout_eval),
        ):
            # Store the calibrated wrapper, including its probability interface.
            # Infer the schema without persisting any training rows as input examples.
            sample = X_train.iloc[:5]
            info = tracking.sklearn.log_model(
                estimator,
                name=f"{family}_calibrated",
                serialization_format="cloudpickle",
                signature=tracking.models.infer_signature(sample, estimator.predict_proba(sample)),
                pyfunc_predict_fn="predict_proba",
            )
            if register_models:
                run = tracking.get_run(tracking.active_run().info.run_id)
                tags = {
                    "model_family": family,
                    "run_id": run.info.run_id,
                    "dataset_sha256": run.data.params["dataset_sha256"],
                    "threshold": str(evaluation["threshold"]),
                    **{
                        f"holdout_{key}": str(evaluation[key])
                        for key in ("f1", "roc_auc", "average_cost")
                        if evaluation[key] is not None
                    },
                }
                tracking.register_model(info.model_uri, f"fraud-risk-{family}", tags=tags)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train calibrated RF + XGBoost fraud models with separate calibration "
            "and holdout test reporting splits."
        )
    )
    p.add_argument("--data", type=Path, required=True, help="Path to CSV with the label column.")
    p.add_argument(
        "--out", type=Path, default=Path("artifacts"), help="Output artifacts directory."
    )
    p.add_argument("--label", type=str, default="Class", help="Label column name.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--calibration-size",
        type=float,
        default=0.15,
        help="Fraction reserved for calibration and threshold selection.",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Fraction reserved for final holdout metric reporting.",
    )
    p.add_argument("--cost-fp", type=float, default=5.0, help="False positive cost.")
    p.add_argument("--cost-fn", type=float, default=500.0, help="False negative cost.")
    p.add_argument(
        "--no-tracking", action="store_true", help="Train without importing or writing to MLflow."
    )
    p.add_argument(
        "--tracking-uri",
        help="MLflow tracking URI (default: MLFLOW_TRACKING_URI or sqlite:///mlflow.db).",
    )
    p.add_argument("--experiment-name", default="fraud-risk-training", help="MLflow experiment.")
    p.add_argument(
        "--register-models",
        action="store_true",
        help="Create RF and XGB model versions in the tracking store; do not assign aliases.",
    )
    args = p.parse_args()
    if args.register_models and args.no_tracking:
        p.error("--register-models requires tracking; remove --no-tracking.")
    return args


def main() -> None:
    args = parse_args()
    train(
        data_path=args.data,
        out_dir=args.out,
        label_col=args.label,
        seed=args.seed,
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
        tracking_enabled=not args.no_tracking,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        register_models=args.register_models,
    )


if __name__ == "__main__":
    main()
