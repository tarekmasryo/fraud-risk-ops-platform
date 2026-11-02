from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from fraud_dashboard.ui.data_io import get_active_dataframe
from fraud_dashboard.ui.pages import (
    DataOverviewPage,
    DataQualitySegmentsPage,
    ModelInsightsPage,
    ModelMetricsPage,
    PredictionEnginePage,
)
from fraud_dashboard.ui.predictors import ui_backend_selector
from fraud_dashboard.ui.theme import init_plotly_theme, inject_css

MODEL_LABELS = {
    "RandomForest (Calibrated)": "rf",
    "XGBoost (Calibrated)": "xgb",
}


def _label_for_model_key(model_key: str) -> str:
    for label, mk in MODEL_LABELS.items():
        if mk == model_key:
            return label
    return model_key


def _pick_default_model(available: list[str]) -> str:
    if "rf" in available:
        return "rf"
    return available[0] if available else "rf"


def _policy_threshold(thresholds: dict, policy_name: str, model_key: str, fallback: float) -> float:
    policies = thresholds.get("policies") or {}
    cfg = policies.get(policy_name) or {}
    values = cfg.get("thresholds") or {}
    try:
        return float(values.get(model_key, fallback))
    except Exception:
        return float(fallback)


def _threshold_defaults(thresholds: dict, model_key: str) -> tuple[float, float]:
    if model_key == "rf":
        balanced = _policy_threshold(
            thresholds, "balanced", "rf", float(thresholds.get("RF_Thr_P90", 0.65))
        )
        min_cost = _policy_threshold(
            thresholds, "min_cost", "rf", float(thresholds.get("RF_Thr_MinCost", 0.07))
        )
        return balanced, min_cost
    balanced = _policy_threshold(
        thresholds, "balanced", "xgb", float(thresholds.get("XGB_Thr_P90", 0.75))
    )
    min_cost = _policy_threshold(
        thresholds, "min_cost", "xgb", float(thresholds.get("XGB_Thr_MinCost", 0.17))
    )
    return balanced, min_cost


@dataclass
class SidebarConfig:
    model_key: str
    thr: float
    policy_name: str | None
    cost_fp: float
    cost_fn: float
    target_precision: float
    target_recall: float
    fp_warn: float
    thr_p90_default: float
    thr_mincost_default: float


class DashboardApp:
    def __init__(self) -> None:
        st.set_page_config(
            page_title="Fraud Risk Ops Console",
            layout="wide",
            page_icon="🧠",
            initial_sidebar_state="expanded",
        )
        init_plotly_theme()
        inject_css()

    def _render_header(self) -> None:
        st.markdown(
            """
<div class="brand-title">
  <h2>🛡️ Fraud Risk Ops Console</h2>
  <div class="brand-sub">Risk decisions · Policy thresholds · Review workflow · Audit signals</div>
</div>
""",
            unsafe_allow_html=True,
        )

    def _render_sidebar(self, predictor) -> SidebarConfig:
        st.sidebar.header("Model")
        available = predictor.available_models()
        if not available:
            available = ["rf", "xgb"]

        label_options = [_label_for_model_key(m) for m in available]
        default_mk = _pick_default_model(available)
        default_label = _label_for_model_key(default_mk)
        idx = max(0, label_options.index(default_label)) if default_label in label_options else 0
        model_label = st.sidebar.selectbox("Select model", label_options, index=idx)
        model_key = MODEL_LABELS.get(model_label, model_label)

        st.sidebar.header("Threshold & Costs")
        thresholds = predictor.thresholds() or {}
        thr_p90_default, thr_mincost_default = _threshold_defaults(thresholds, model_key)

        cost_fp = st.sidebar.number_input(
            "Cost of False Positive (COST_FP)",
            min_value=0.0,
            max_value=10000.0,
            value=float(thresholds.get("COST_FP", 5.0)),
            step=1.0,
        )
        cost_fn = st.sidebar.number_input(
            "Cost of False Negative (COST_FN)",
            min_value=0.0,
            max_value=100000.0,
            value=float(thresholds.get("COST_FN", 200.0)),
            step=5.0,
        )

        policies = thresholds.get("policies") or {}
        policy_options = [p for p in ["strict", "balanced", "min_cost", "lenient"] if p in policies]
        if not policy_options:
            policy_options = ["balanced"]

        st.sidebar.markdown("**Operating policy**")
        threshold_mode = st.sidebar.radio(
            "Threshold mode",
            ["Policy preset", "Manual threshold"],
            horizontal=True,
        )
        policy_name: str | None = None
        if threshold_mode == "Policy preset":
            default_policy_idx = (
                policy_options.index("balanced") if "balanced" in policy_options else 0
            )
            policy_name = st.sidebar.selectbox(
                "Policy preset",
                policy_options,
                index=default_policy_idx,
                format_func=lambda value: value.replace("_", " ").title(),
            )
            thr = _policy_threshold(thresholds, policy_name, model_key, float(thr_p90_default))
            st.sidebar.caption(f"Resolved threshold: {thr:.3f}")
        else:
            thr = st.sidebar.slider(
                "Manual decision threshold",
                0.0,
                1.0,
                float(thr_p90_default),
                0.001,
            )

        st.sidebar.markdown("---")
        st.sidebar.markdown("**Threshold Finder**")
        target_precision = st.sidebar.slider("Target Precision", 0.50, 0.99, 0.90, 0.01)
        target_recall = st.sidebar.slider("Target Recall", 0.50, 0.99, 0.80, 0.01)
        fp_warn = st.sidebar.number_input(
            "FP warning threshold", min_value=0.0, max_value=100000.0, value=2000.0, step=100.0
        )

        return SidebarConfig(
            model_key=model_key,
            thr=float(thr),
            policy_name=policy_name,
            cost_fp=float(cost_fp),
            cost_fn=float(cost_fn),
            target_precision=float(target_precision),
            target_recall=float(target_recall),
            fp_warn=float(fp_warn),
            thr_p90_default=float(thr_p90_default),
            thr_mincost_default=float(thr_mincost_default),
        )

    def run(self) -> None:
        self._render_header()

        predictor, _backend_meta = ui_backend_selector()
        features = predictor.schema_features()
        if not features:
            # fallback: creditcard-like features
            features = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

        cfg = self._render_sidebar(predictor)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📊 Data Overview",
                "⚙️ Prediction Engine",
                "📈 Model Metrics",
                "🧠 Model Insights",
                "🧹 Data Quality & Segments",
            ]
        )

        with tab1:
            uploaded = st.file_uploader(
                "Upload a CSV (optional)", type=["csv", "gz"], key="upload_tab1"
            )
            df, msg = get_active_dataframe(uploaded)
            DataOverviewPage(expected_features=features).render(df=df, src_msg=msg)

        with tab2:
            uploaded_pred = st.file_uploader(
                "Upload a CSV for prediction (optional)", type=["csv", "gz"], key="upload_tab2"
            )
            df_pred, msg_pred = get_active_dataframe(uploaded_pred)
            PredictionEnginePage(predictor=predictor, expected_features=features).render(
                df_pred=df_pred,
                src_msg=msg_pred,
                model_key=cfg.model_key,
                thr=cfg.thr,
                policy_name=cfg.policy_name,
                cost_fp=cfg.cost_fp,
                cost_fn=cfg.cost_fn,
            )

        with tab3:
            ModelMetricsPage(
                target_precision=cfg.target_precision,
                target_recall=cfg.target_recall,
                fp_warn=cfg.fp_warn,
                thr_p90_default=cfg.thr_p90_default,
                thr_mincost_default=cfg.thr_mincost_default,
            ).render(thr=cfg.thr, cost_fp=cfg.cost_fp, cost_fn=cfg.cost_fn)

        with tab4:
            ModelInsightsPage(top_k=20).render()

        with tab5:
            uploaded_profile = st.file_uploader(
                "Upload a CSV for profiling (optional)", type=["csv", "gz"], key="upload_tab5"
            )
            if uploaded_profile is not None:
                df_prof, prof_msg = get_active_dataframe(uploaded_profile)
            else:
                last = st.session_state.get("last_run", None)
                if last is not None:
                    df_prof, prof_msg = last.df, "Using last prediction dataset."
                else:
                    df_prof, prof_msg = get_active_dataframe(None)

            DataQualitySegmentsPage().render(df=df_prof, src_msg=prof_msg, thr=cfg.thr)

        st.markdown("---")
        st.caption("Fraud Risk Ops · Calibrated RF/XGB · © Tarek Masryo")


def main() -> None:
    DashboardApp().run()
