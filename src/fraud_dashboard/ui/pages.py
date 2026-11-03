from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from fraud_dashboard.core.artifacts import load_models
from fraud_dashboard.ui.api_client import BackendUnavailableError
from fraud_dashboard.ui.data_io import validate_ui_dataframe
from fraud_dashboard.ui.predictors import render_backend_unavailable
from fraud_dashboard.ui.theme import PRIMARY, style_fig


def _has_param(fn, name: str) -> bool:
    """Return True if callable `fn` has a parameter named `name`.

    This guards against environments where `inspect.signature` may fail.
    """
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def dataframe_full_width(container, df: pd.DataFrame) -> None:
    """Render a dataframe using the widest supported Streamlit API.

    Streamlit changed sizing APIs across versions:
    - Older: use_container_width=True
    - Newer: width="stretch"
    """
    fn = container.dataframe

    if _has_param(fn, "width"):
        fn(df, width="stretch")
        return

    if _has_param(fn, "use_container_width"):
        fn(df, use_container_width=True)
        return

    fn(df)


def plotly_full_width(container, fig) -> None:
    """Render a Plotly figure full-width across Streamlit versions."""
    fn = container.plotly_chart

    if _has_param(fn, "width"):
        fn(fig, width="stretch")
        return

    if _has_param(fn, "use_container_width"):
        fn(fig, use_container_width=True)
        return

    fn(fig)


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Amount" in out.columns and "_log_amount" not in out.columns:
        out["_log_amount"] = np.log1p(out["Amount"].astype(float))
    if "Time" in out.columns and "Hour_from_start_mod24" not in out.columns:
        hours = (out["Time"].astype(float) / 3600.0) % 24.0
        out["Hour_from_start_mod24"] = hours.astype(int)
    return out


def expected_cost(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float, cost_fp: float, cost_fn: float
) -> float:
    y_pred = (y_prob >= thr).astype(int)
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return fp * float(cost_fp) + fn * float(cost_fn)


@dataclass
class RunState:
    df: pd.DataFrame
    X: pd.DataFrame
    features: list[str]
    y_true: np.ndarray | None
    y_prob: np.ndarray
    thr: float
    cost_fp: float
    cost_fn: float
    model_key: str
    policy_name: str | None
    elapsed_s: float


def prediction_results_frame(
    df: pd.DataFrame,
    probs: np.ndarray,
    *,
    threshold: float,
    policy: str | None,
    details: list[dict[str, Any]] | None,
    local_policy_version: str | None = None,
) -> pd.DataFrame:
    """Keep backend decisions as supplied, separate from transaction input fields."""
    result = pd.DataFrame(
        {"fraud_proba": probs, "fraud_pred": (probs >= threshold).astype(int)}, index=df.index
    )
    if details is not None:
        if len(details) != len(df):
            raise RuntimeError("API result details do not match the input row count.")
        returned = pd.DataFrame(details, index=df.index)
        if "label" in returned:
            result["fraud_pred"] = returned["label"]
        for key in (
            "decision",
            "policy",
            "policy_version",
            "threshold",
            "review_required",
            "risk_band",
            "request_id",
        ):
            if key in returned:
                result[key] = returned[key]
    else:
        result["policy"] = policy or "manual_override"
        result["policy_version"] = local_policy_version or "Not supplied"
        result["threshold"] = threshold
    return result


class DataOverviewPage:
    def __init__(self, *, expected_features: list[str]) -> None:
        self.expected_features = expected_features

    def render(self, *, df: pd.DataFrame, src_msg: str) -> None:
        st.subheader("Dataset")
        st.info(src_msg)

        if df.empty:
            st.warning("No data loaded yet.")
            return
        try:
            df = validate_ui_dataframe(df)
        except ValueError as exc:
            st.error(str(exc))
            return

        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        dataframe_full_width(st, df.head(50))

        if "Class" in df.columns:
            pos = int(df["Class"].sum())
            total = len(df)
            st.write(f"Class distribution: Fraud = {pos:,} / {total:,} ({pos / total:.4%})")

        schema_df = pd.DataFrame({"expected_columns": self.expected_features})
        st.download_button(
            "Download expected schema (columns).csv",
            schema_df.to_csv(index=False).encode("utf-8"),
            file_name="expected_schema.csv",
            mime="text/csv",
        )


class PredictionEnginePage:
    def __init__(self, *, predictor, expected_features: list[str]) -> None:
        self.predictor = predictor
        self.expected_features = expected_features

    def _ensure_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        df2 = build_engineered_features(validate_ui_dataframe(df))
        missing = [c for c in self.expected_features if c not in df2.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing[:12]}{' ...' if len(missing) > 12 else ''}"
            )
        return df2[self.expected_features].copy(), self.expected_features

    def render(
        self,
        *,
        df_pred: pd.DataFrame,
        src_msg: str,
        model_key: str,
        thr: float,
        policy_name: str | None,
        cost_fp: float,
        cost_fn: float,
    ) -> None:
        st.subheader("Run Batch Prediction")
        st.info(src_msg)
        st.session_state.pop("last_run", None)

        if df_pred.empty:
            st.warning("No data to predict on yet.")
            return

        try:
            df_pred = validate_ui_dataframe(df_pred)
            X, used_cols = self._ensure_features(df_pred)
        except ValueError as exc:
            st.error(str(exc))
            return
        if len(used_cols) == 0:
            st.error("No usable feature columns found.")
            return

        try:
            with st.spinner("Running inference..."):
                probs, elapsed = self.predictor.predict_proba_batch(
                    X,
                    model_key=model_key,
                    threshold=thr if policy_name is None else None,
                    policy=policy_name,
                )
        except BackendUnavailableError:
            render_backend_unavailable()
            return

        details = getattr(self.predictor, "last_results", None)
        details = details if isinstance(details, list) else None
        thresholds = self.predictor.thresholds()
        version = thresholds.get("artifact_version") if isinstance(thresholds, dict) else None
        results = prediction_results_frame(
            df_pred,
            probs,
            threshold=thr,
            policy=policy_name,
            details=details,
            local_policy_version=version if isinstance(version, str) else None,
        )
        preds = results["fraud_pred"].to_numpy()
        raw = df_pred.drop(columns=results.columns, errors="ignore").rename(
            columns={
                key: f"input_{key}"
                for key in (
                    "decision",
                    "policy",
                    "policy_version",
                    "threshold",
                    "review_required",
                    "risk_band",
                    "request_id",
                )
            }
        )
        out = pd.concat([results, raw], axis=1)
        y_true = df_pred["Class"].values if "Class" in df_pred.columns else None

        policy_label = policy_name or "manual_override"
        st.caption(f"Operating point: {policy_label} · threshold={thr:.3f}")
        st.markdown("### Prediction results")
        st.caption(
            "Risk score (fraud_proba) is a model probability; fraud_pred is the binary fraud "
            "label. Operational decisions are shown only when supplied by the backend."
        )
        for key, label in (("policy", "Policy"), ("policy_version", "Policy version")):
            values = results[key].dropna().astype(str).unique() if key in results else []
            st.write(f"{label}: {', '.join(values) if len(values) else 'Not supplied'}")
        if "decision" not in results:
            st.info(
                "Operational decisions were not supplied; no approve/review decision is inferred."
            )

        if y_true is not None:
            prec = precision_score(y_true, preds, zero_division=0)
            rec = recall_score(y_true, preds, zero_division=0)
            f1 = f1_score(y_true, preds, zero_division=0)
            cost_now = expected_cost(y_true, probs, thr, float(cost_fp), float(cost_fn))
            st.success(
                f"Inference: {elapsed:.3f}s — Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | Expected Cost: {cost_now:,.0f}"
            )
        else:
            st.info(f"Inference: {elapsed:.3f}s — Labels not found; metrics skipped.")

        st.session_state["last_run"] = RunState(
            df=df_pred.copy(deep=True),
            X=X,
            features=used_cols,
            y_true=y_true,
            y_prob=probs,
            thr=float(thr),
            cost_fp=float(cost_fp),
            cost_fn=float(cost_fn),
            model_key=model_key,
            policy_name=policy_name,
            elapsed_s=float(elapsed),
        )

        preview_all = st.checkbox("Return all rows (preview full table)", value=False)
        preview = results if preview_all else results.head(50)
        dataframe_full_width(st, preview)
        with st.expander("Raw transaction features", expanded=False):
            dataframe_full_width(st, df_pred if preview_all else df_pred.head(50))
        st.download_button(
            "Download predictions.csv",
            out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )


class ModelMetricsPage:
    def __init__(
        self,
        *,
        target_precision: float,
        target_recall: float,
        fp_warn: float,
        thr_p90_default: float,
        thr_mincost_default: float,
    ) -> None:
        self.target_precision = target_precision
        self.target_recall = target_recall
        self.fp_warn = fp_warn
        self.thr_p90_default = thr_p90_default
        self.thr_mincost_default = thr_mincost_default

    def render(self, *, thr: float, cost_fp: float, cost_fn: float) -> None:
        st.subheader("Threshold & Cost Analysis")
        state = st.session_state.get("last_run", None)
        if not state or state.y_true is None:
            st.info(
                "Upload a labeled dataset (with 'Class') and run prediction to view model metrics."
            )
            return

        y_true = state.y_true
        y_prob = state.y_prob
        if np.unique(y_true).size < 2:
            st.info(
                "Evaluation requires both Class=0 and Class=1. ROC/AUC and threshold comparison are unavailable for a single-class dataset."
            )
            return
        y_pred = (y_prob >= thr).astype(int)

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        cost_now = expected_cost(y_true, y_prob, thr, float(cost_fp), float(cost_fn))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Precision", f"{prec:.3f}")
        k2.metric("Recall", f"{rec:.3f}")
        k3.metric("F1", f"{f1:.3f}")
        k4.metric("Expected Cost", f"{cost_now:,.0f}")
        k5.metric("True Positives", f"{tp:,}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### KPI Trends (across thresholds)")
        s1, s2, s3 = st.columns(3)
        grid_spark = np.linspace(0.0, 1.0, 51)

        def kpi_sparkline(values, title, fmt="{:.3f}") -> None:
            fig = go.Figure(go.Scatter(y=values, mode="lines", fill="tozeroy"))
            fig.update_layout(height=80, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            fig = style_fig(fig, None)
            st.metric(title, fmt.format(values[-1]))
            plotly_full_width(st, fig)

        prec_list, rec_list, cost_list = [], [], []
        for t in grid_spark:
            y_hat = (y_prob >= t).astype(int)
            prec_list.append(precision_score(y_true, y_hat, zero_division=0))
            rec_list.append(recall_score(y_true, y_hat, zero_division=0))
            cost_list.append(expected_cost(y_true, y_prob, t, float(cost_fp), float(cost_fn)))

        with s1:
            kpi_sparkline(prec_list, "Precision (sweep)")
        with s2:
            kpi_sparkline(rec_list, "Recall (sweep)")
        with s3:
            kpi_sparkline(cost_list, "Expected Cost (sweep)", fmt="{:,.0f}")

        st.markdown("---")
        c1, c2, c3 = st.columns(3)

        warn_color = "red" if fp >= self.fp_warn else PRIMARY

        cm = np.array([[tn, fp], [fn, tp]])
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Pred: Non-Fraud", "Pred: Fraud"],
                y=["Actual: Non-Fraud", "Actual: Fraud"],
                text=cm,
                texttemplate="%{text}",
                colorscale="RdBu",
                reversescale=True,
                showscale=False,
            )
        )
        fig_cm.update_xaxes(side="top")
        fig_cm = style_fig(fig_cm, "Confusion Matrix")
        fig_cm.add_annotation(
            text=f"FP={fp:,} (warn≥{int(self.fp_warn)})",
            x=0.95,
            y=-0.18,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color=warn_color, size=12),
        )
        plotly_full_width(c1, fig_cm)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})"))
        fig_roc.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash"))
        )
        fig_roc.update_xaxes(title="False Positive Rate")
        fig_roc.update_yaxes(title="True Positive Rate")
        fig_roc = style_fig(fig_roc, "ROC Curve")
        plotly_full_width(c2, fig_roc)

        precision_arr, recall_arr, thr_arr = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_arr, precision_arr)
        fig_pr = go.Figure()
        fig_pr.add_trace(
            go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name=f"PR (AUC={pr_auc:.3f})")
        )
        if thr_arr.size > 0:
            idx_op = int(np.argmin(np.abs(thr_arr - thr)))
            op_p = float(precision_arr[idx_op])
            op_r = float(recall_arr[idx_op])
        else:
            op_p, op_r = float(prec), float(rec)
        fig_pr.add_trace(
            go.Scatter(
                x=[op_r],
                y=[op_p],
                mode="markers",
                marker=dict(size=10, line=dict(width=1), color=PRIMARY),
                name=f"Operating @ thr={thr:.3f}",
            )
        )
        fig_pr.update_xaxes(title="Recall")
        fig_pr.update_yaxes(title="Precision")
        fig_pr = style_fig(fig_pr, "Precision–Recall Curve")
        plotly_full_width(c3, fig_pr)

        st.markdown("---")
        grid = np.linspace(0.0, 1.0, 201)
        costs = [expected_cost(y_true, y_prob, t, float(cost_fp), float(cost_fn)) for t in grid]

        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(x=grid, y=costs, mode="lines", name="Expected Cost"))
        cur_cost = expected_cost(y_true, y_prob, thr, float(cost_fp), float(cost_fn))
        fig_cost.add_vline(
            x=thr,
            line_dash="dash",
            line_color=warn_color,
            annotation_text=f"thr={thr:.3f}<br>FP={fp:,}<br>cost={cur_cost:,.0f}",
            annotation_position="top right",
        )
        fig_cost.add_vline(
            x=self.thr_p90_default,
            line_color="green",
            annotation_text=f"P@90%≈{self.thr_p90_default:.3f}",
            line_dash="dot",
        )
        fig_cost.add_vline(
            x=self.thr_mincost_default,
            line_color="orange",
            annotation_text=f"MinCost≈{self.thr_mincost_default:.3f}",
            line_dash="dot",
        )
        fig_cost.update_layout(xaxis_title="Threshold", yaxis_title="Expected Cost")
        fig_cost = style_fig(fig_cost, "Expected Cost vs Threshold")
        plotly_full_width(st, fig_cost)

        cta1, cta2, cta3 = st.columns(3)
        if cta1.button("Find Min-Cost Threshold"):
            idx = int(np.argmin(costs))
            best_thr = float(grid[idx])
            st.info(f"Min-Cost threshold ≈ {best_thr:.3f} (Expected Cost={costs[idx]:,.0f})")

        if cta2.button("Find Threshold @ Target Precision"):
            feasible: list[tuple[float, float]] = []
            for t in grid:
                y_hat = (y_prob >= t).astype(int)
                feasible.append((float(t), precision_score(y_true, y_hat, zero_division=0)))
            good = [t for t, p in feasible if p >= self.target_precision]
            st.info(
                f"Threshold @ Precision ≥ {self.target_precision:.2f} → {min(good):.3f}"
                if good
                else "No threshold achieves the requested precision."
            )

        if cta3.button("Find Threshold @ Target Recall"):
            feasible_r: list[tuple[float, float]] = []
            for t in grid:
                y_hat = (y_prob >= t).astype(int)
                feasible_r.append((float(t), recall_score(y_true, y_hat, zero_division=0)))
            good_r = [t for t, r in feasible_r if r >= self.target_recall]
            st.info(
                f"Threshold @ Recall ≥ {self.target_recall:.2f} → {max(good_r):.3f}"
                if good_r
                else "No threshold achieves the requested recall."
            )

        if st.button("Animate Threshold Sweep (PR)"):
            frames = []
            thr_sweep = np.linspace(0, 1, 30)
            for t in thr_sweep:
                if thr_arr.size > 0:
                    idx_t = int(np.argmin(np.abs(thr_arr - t)))
                    r_t = float(recall_arr[idx_t])
                    p_t = float(precision_arr[idx_t])
                else:
                    yhat_t = (y_prob >= t).astype(int)
                    p_t = float(precision_score(y_true, yhat_t, zero_division=0))
                    r_t = float(recall_score(y_true, yhat_t, zero_division=0))
                frames.append(
                    go.Frame(
                        data=[
                            go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name="PR"),
                            go.Scatter(
                                x=[r_t],
                                y=[p_t],
                                mode="markers",
                                marker=dict(size=10),
                                name=f"thr={t:.2f}",
                            ),
                        ],
                        name=f"{t:.2f}",
                    )
                )

            fig_anim = go.Figure(
                data=[
                    go.Scatter(x=recall_arr, y=precision_arr, mode="lines", name="PR"),
                    go.Scatter(x=[op_r], y=[op_p], mode="markers", name=f"thr={thr:.2f}"),
                ],
                frames=frames,
            )
            fig_anim.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                updatemenus=[
                    dict(
                        type="buttons", buttons=[dict(label="Play", method="animate", args=[None])]
                    )
                ],
            )
            fig_anim = style_fig(fig_anim, "PR Curve · Threshold Sweep")
            plotly_full_width(st, fig_anim)


class ModelInsightsPage:
    def __init__(self, *, top_k: int = 20) -> None:
        self.top_k = top_k

    @st.cache_resource(show_spinner=False)
    def _models(_self) -> dict[str, Any]:
        """Load local models once per session.

        Streamlit caches function calls by hashing all arguments.
        Instance methods include ``self`` which is not hashable by default.
        Prefixing the argument name with an underscore tells Streamlit to
        ignore it for caching purposes.
        """
        return load_models()

    def render(self) -> None:
        st.subheader("Feature Importance (Permutation)")
        state = st.session_state.get("last_run", None)
        if not state or state.y_true is None:
            st.info("Run a labeled prediction first to compute importances.")
            return

        X = state.X
        y_true = state.y_true
        model_key = state.model_key

        if np.unique(y_true).size < 2:
            st.info("ROC AUC permutation importance requires both Class=0 and Class=1.")
            return

        if model_key not in self._models():
            st.warning(f"Local model '{model_key}' not found. Cannot compute importances.")
            return

        # Keep it responsive
        X_sub, y_sub = X, y_true
        if len(X_sub) > 6000:
            idx = np.random.RandomState(42).choice(len(X_sub), 6000, replace=False)
            X_sub = X_sub.iloc[idx]
            y_sub = y_sub[idx]

        if np.unique(y_sub).size < 2:
            st.info(
                "The evaluation sample contains only one class; permutation importance is unavailable."
            )
            return

        with st.spinner("Computing permutation importances..."):
            res = permutation_importance(
                self._models()[model_key],
                X_sub,
                y_sub,
                n_repeats=5,
                random_state=42,
                scoring="roc_auc",
            )
            imp = (
                pd.DataFrame(
                    {
                        "feature": X_sub.columns,
                        "importance_mean": res.importances_mean,
                        "importance_std": res.importances_std,
                    }
                )
                .sort_values("importance_mean", ascending=False)
                .head(self.top_k)
            )

        fig_imp = px.bar(
            imp,
            x="importance_mean",
            y="feature",
            error_x="importance_std",
            orientation="h",
            title=None,
        )
        fig_imp.update_traces(marker_line_width=0.5, opacity=0.95)
        fig_imp = style_fig(fig_imp, f"Top {self.top_k} Feature Importances (ROC AUC drop)")
        plotly_full_width(st, fig_imp)
        st.caption("Interpret with care (correlated features may share importance).")


class DataQualitySegmentsPage:
    def __init__(self) -> None:
        pass

    def render(self, *, df: pd.DataFrame, src_msg: str, thr: float) -> None:
        st.subheader("Profiling & Segmented Performance")
        st.info(src_msg)

        state = st.session_state.get("last_run", None)

        if df.empty:
            st.warning("No data available for profiling yet.")
            return
        try:
            df = validate_ui_dataframe(df)
        except ValueError as exc:
            st.error(str(exc))
            return
        matches = state is not None and df.equals(state.df)
        y_true = state.y_true if matches else None
        y_prob = state.y_prob if matches else None
        if state is not None and not matches:
            st.info(
                "Segmented prediction metrics are unavailable: this dataset differs from the last prediction dataset. Run prediction on this dataset first."
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Rows", f"{len(df):,}")
        with c2:
            st.metric("Columns", f"{df.shape[1]:,}")
        with c3:
            st.metric("Missing values", f"{int(df.isna().sum().sum()):,}")

        st.markdown("### Key Distributions")
        col1, col2 = st.columns(2)

        plot_df = df
        if len(df) > 120_000:
            plot_df = df.sample(120_000, random_state=42)

        if "Amount" in df.columns:
            h1 = px.histogram(plot_df, x="Amount", nbins=60, marginal="box", title=None)
            h1.update_traces(opacity=0.9)
            h1 = style_fig(h1, "Amount Distribution")
            plotly_full_width(col1, h1)

        if "Time" in df.columns:
            df_tmp = plot_df.copy()
            if "Hour_from_start_mod24" not in df_tmp.columns:
                df_tmp = build_engineered_features(df_tmp)
            h2 = px.histogram(df_tmp, x="Hour_from_start_mod24", nbins=24, title=None)
            h2 = style_fig(h2, "Hour from start (mod 24)")
            plotly_full_width(col2, h2)
            st.caption(
                "Time is elapsed seconds since the first transaction. These buckets do not represent local time, business hours or night-time activity."
            )

        if y_true is not None and y_prob is not None and "Amount" in df.columns:
            st.markdown("### Segmented Metrics by Amount (quintiles)")
            seg = pd.DataFrame({"Amount": df["Amount"].values, "y_true": y_true, "y_prob": y_prob})
            if seg["Amount"].nunique() > 1:
                seg["bin"] = pd.qcut(seg["Amount"], q=5, duplicates="drop")
                rows: list[dict[str, Any]] = []
                for b, sub in seg.groupby("bin", observed=True):
                    yb = sub["y_true"].values
                    pb = sub["y_prob"].values
                    yhat = (pb >= thr).astype(int)
                    rows.append(
                        {
                            "Amount bin": str(b),
                            "Fraud rate": float(np.mean(yb)),
                            "Predicted fraud rate": float(np.mean(yhat)),
                            "Precision": precision_score(yb, yhat, zero_division=0),
                            "Recall": recall_score(yb, yhat, zero_division=0),
                            "F1": f1_score(yb, yhat, zero_division=0),
                            "Count": int(len(sub)),
                        }
                    )
                dataframe_full_width(st, pd.DataFrame(rows))
            else:
                st.info("Skipping segmented metrics: 'Amount' has a single unique value.")
