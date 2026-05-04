"""Professional Streamlit dashboard for AI network performance optimization."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

try:
    import shap
except ImportError:
    shap = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_data
from src.features import create_features
from src.predict import Predictor


MODEL_PATH = "models/network_model.pkl"
MAX_RECORDS = 50_000
VIZ_SAMPLE_SIZE = 10_000
SHAP_SAMPLE_SIZE = 500
FEATURE_COLUMNS = ["latency", "throughput", "packet_loss"]


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")


def style_dark_axes(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Make matplotlib charts readable on Streamlit dark themes."""
    ax.set_facecolor("#0f172a")
    ax.figure.set_facecolor("#0f172a")
    ax.tick_params(colors="#e5e7eb")
    ax.grid(color="#334155", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#475569")
    if xlabel:
        ax.set_xlabel(xlabel, color="#e5e7eb")
    if ylabel:
        ax.set_ylabel(ylabel, color="#e5e7eb")
    ax.title.set_color("#f8fafc")


def style_colorbar(colorbar, label: str) -> None:
    """Style matplotlib colorbars for dark UI."""
    colorbar.set_label(label, color="#e5e7eb")
    colorbar.ax.yaxis.set_tick_params(color="#e5e7eb", labelcolor="#e5e7eb")
    colorbar.outline.set_edgecolor("#475569")


def clean_plot_df(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Prepare plot data without relying on deprecated pandas options."""
    plot_df = df.replace([np.inf, -np.inf], np.nan)
    return plot_df.dropna(subset=subset)


def clip_for_plot(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Clip extreme tails for readable density plots without changing source data."""
    plot_df = df.copy()
    for column in columns:
        upper = plot_df[column].quantile(0.99)
        plot_df[column] = np.clip(plot_df[column], 0, upper)
    return plot_df


def prepare_visual_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create readable clipped and log-scaled features for plots."""
    plot_df = clean_plot_df(df, FEATURE_COLUMNS).copy()
    plot_df["latency"] = np.clip(plot_df["latency"], 0, plot_df["latency"].quantile(0.99))
    plot_df["throughput"] = np.clip(plot_df["throughput"], 0, plot_df["throughput"].quantile(0.99))
    plot_df["packet_loss"] = np.clip(plot_df["packet_loss"], 0, 1)
    plot_df["log_latency"] = np.log1p(plot_df["latency"])
    plot_df["log_throughput"] = np.log1p(plot_df["throughput"])

    for column in ["congestion", "tower", "time", "Label"]:
        if column in df.columns:
            plot_df[column] = df.loc[plot_df.index, column]

    return plot_df


def kpi_card(label: str, value: str, color: str) -> None:
    """Render a compact colored KPI card."""
    st.markdown(
        f"""
        <div style="
            border-left: 6px solid {color};
            background: #ffffff;
            border-radius: 10px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 1px 8px rgba(15, 23, 42, 0.08);
            min-height: 112px;
        ">
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">{label}</div>
            <div style="font-size: 1.65rem; color: #0f172a; font-weight: 700; margin-top: 10px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Loading real CICIDS data...")
def load_cached_data() -> pd.DataFrame:
    """Load real raw CICIDS data with Streamlit caching."""
    return load_data()


def get_data() -> pd.DataFrame:
    """Compatibility wrapper for cached raw data loading."""
    return load_cached_data()


@st.cache_data(show_spinner="Creating dashboard features...")
def prepare_dashboard_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create features and cache the engineered dashboard frame."""
    df = raw_df.copy()
    df = create_features(df)

    for column in FEATURE_COLUMNS + ["congestion"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + ["congestion"])
    keep_columns = FEATURE_COLUMNS + ["congestion"]
    if "Label" in df.columns:
        keep_columns.append("Label")
    df = df[keep_columns]

    if len(df) > MAX_RECORDS:
        df = df.sample(MAX_RECORDS, random_state=42)

    df = df.reset_index(drop=True)
    df["time"] = range(len(df))
    df["tower"] = create_tower_segments(df["throughput"])
    return df


def load_dashboard_data() -> pd.DataFrame:
    """Compatibility wrapper for existing dashboard code."""
    return prepare_dashboard_data(get_data())


@st.cache_data
def sample_for_visualization(df: pd.DataFrame, sample_size: int = VIZ_SAMPLE_SIZE) -> pd.DataFrame:
    """Create a stable lightweight visualization sample."""
    if len(df) <= sample_size:
        return df.copy()
    return df.sample(sample_size, random_state=42).reset_index(drop=True)


@st.cache_data
def compute_dashboard_metrics(df: pd.DataFrame) -> dict:
    """Precompute reusable dashboard metrics outside tab rendering."""
    return {
        "rows": len(df),
        "features": len(FEATURE_COLUMNS),
        "missing_values": int(df[FEATURE_COLUMNS + ["congestion"]].isna().sum().sum()),
        "congestion_rate": float(df["congestion"].mean() * 100),
        "avg_latency": float(df["latency"].mean()),
        "avg_throughput": float(df["throughput"].mean()),
        "latency_threshold": float(df["latency"].quantile(0.75)),
        "throughput_threshold": float(df["throughput"].quantile(0.25)),
    }


@st.cache_data
def compute_model_metrics(_model, df: pd.DataFrame) -> dict:
    """Evaluate the loaded model on test set only (avoiding overfitting assessment).
    
    Uses 80/20 train/test split with random_state=42 to match training pipeline.
    Evaluates metrics ONLY on test set for realistic generalization performance.
    """
    # Clean data first
    eval_df = clean_plot_df(df, FEATURE_COLUMNS + ["congestion"]).copy()
    
    # Perform 80/20 split to get test set (matches training split)
    X_train, X_test, y_train, y_test = train_test_split(
        eval_df[FEATURE_COLUMNS],
        eval_df["congestion"],
        test_size=0.2,
        random_state=42,
        stratify=eval_df["congestion"]
    )
    
    # Reconstruct test dataframe with features and target
    test_df = X_test.copy()
    test_df["congestion"] = y_test.values
    
    # Limit test set size for visualization
    if len(test_df) > VIZ_SAMPLE_SIZE:
        test_df = test_df.sample(VIZ_SAMPLE_SIZE, random_state=42)

    y_true = test_df["congestion"].astype(int)
    y_pred = _model.predict(test_df[FEATURE_COLUMNS])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


@st.cache_data
def compute_tower_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rank towers by observed congestion rate."""
    tower_summary = (
        df.groupby("tower", observed=False)
        .agg(records=("congestion", "size"), congestion_rate=("congestion", "mean"))
        .reset_index()
    )
    tower_summary["congestion_percentage"] = tower_summary["congestion_rate"] * 100
    tower_summary = tower_summary.sort_values("congestion_percentage", ascending=False).reset_index(drop=True)
    tower_summary["rank"] = tower_summary.index + 1
    return tower_summary


@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    """Load the trained model artifact once."""
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Preparing predictor...")
def load_predictor() -> Predictor:
    """Create a Predictor around the cached model without retraining."""
    predictor = Predictor.__new__(Predictor)
    predictor.model = load_model()
    return predictor


def create_tower_segments(throughput: pd.Series) -> pd.Series:
    """Create three tower segments from real throughput values."""
    try:
        return pd.qcut(throughput, 3, labels=["A", "B", "C"])
    except ValueError:
        ranked_throughput = throughput.rank(method="first")
        return pd.qcut(ranked_throughput, 3, labels=["A", "B", "C"])


def get_model_feature_importance(predictor: Predictor) -> pd.DataFrame:
    """Extract feature importance from the trained model."""
    model = predictor.model
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    return (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def normalize_shap_values(shap_values):
    """Return class-1 SHAP values across SHAP/sklearn output variants."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    return values


@st.cache_data(show_spinner="Computing SHAP explanations...")
def compute_shap_values(_model, x_sample: pd.DataFrame):
    """Compute cached SHAP values for a small model-ready sample."""
    if shap is None:
        return None, None, None
    explainer = shap.TreeExplainer(_model)
    sample = x_sample.sample(min(len(x_sample), 100), random_state=42)
    shap_values = explainer.shap_values(sample)
    return explainer.expected_value, normalize_shap_values(shap_values), sample


def show_feature_importance_fallback(model) -> None:
    """Show feature importance when SHAP cannot render."""
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0f172a")
        ax.barh(importance["feature"], importance["importance"], color="#38bdf8")
        style_dark_axes(ax, xlabel="Importance")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Feature importance is not available for this model.")


def show_shap_summary(model, x_sample: pd.DataFrame) -> None:
    """Render a compact SHAP summary plot when SHAP is available."""
    if shap is None:
        st.info("SHAP is not installed. Run `pip install shap` to enable explainability plots.")
        return

    try:
        _, shap_values, sample = compute_shap_values(model, x_sample)
        if shap_values is None:
            st.info("SHAP explanations are unavailable for the loaded model.")
            show_feature_importance_fallback(model)
            return

        plt.figure(figsize=(8, 4.8), facecolor="#0f172a")
        shap.summary_plot(shap_values, sample, show=False, plot_type="bar", color="#38bdf8")
        fig = plt.gcf()
        for ax in fig.axes:
            style_dark_axes(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.warning("SHAP not available, showing feature importance instead")
        st.caption(str(e))
        show_feature_importance_fallback(model)


def show_shap_force_plot(model, input_row: pd.DataFrame) -> None:
    """Render local SHAP force-style explanation for a single prediction."""
    if shap is None:
        st.info("SHAP is not installed. Run `pip install shap` to enable local explanations.")
        return

    try:
        expected_value, shap_values, sample = compute_shap_values(model, input_row)
        if shap_values is None:
            st.info("SHAP explanations are unavailable for the loaded model.")
            show_feature_importance_fallback(model)
            return

        base_value = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1 else expected_value
        force_plot = shap.force_plot(base_value, shap_values[0], sample.iloc[0])
        components.html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=260,
            scrolling=True,
        )
    except Exception as e:
        st.warning("SHAP not available, showing feature importance instead")
        st.caption(str(e))
        show_feature_importance_fallback(model)


def readable_feature_name(feature: str) -> str:
    """Convert model feature names into dashboard text."""
    names = {
        "latency": "Latency",
        "throughput": "Throughput imbalance",
        "packet_loss": "Packet loss",
    }
    return names.get(feature, feature.replace("_", " "))


def alert_from_probability(probability: float) -> str:
    """Map congestion probability to an early warning state."""
    if probability < 0.4:
        return "Normal"
    if probability <= 0.7:
        return "Warning"
    return "Critical"


def show_alert(probability: float) -> None:
    """Render probability-based warning state."""
    level = alert_from_probability(probability)
    message = f"{level} network state | Congestion probability: {probability:.1%}"

    if level == "Normal":
        st.success(message)
    elif level == "Warning":
        st.warning(message)
    else:
        st.error(message)


def dynamic_suggestions(
    df: pd.DataFrame,
    latency: float,
    throughput: float,
    packet_loss: float,
    congestion_prob: float = 0.0,
) -> list[str]:
    """Generate input-specific optimization actions."""
    suggestions = []
    latency_threshold = float(df["latency"].quantile(0.75))
    throughput_threshold = float(df["throughput"].quantile(0.25))

    if packet_loss > 0.5:
        suggestions.append("Check cables, hardware, and reduce packet drops")
    if latency > latency_threshold:
        suggestions.append("Optimize routing paths or reduce hops")
    if throughput < throughput_threshold:
        suggestions.append("Increase bandwidth or load balance traffic")
    if congestion_prob > 0.9:
        suggestions.append("Immediate load balancing required")

    if not suggestions:
        suggestions.append("System operating normally")

    return suggestions


def input_root_causes(df: pd.DataFrame, latency: float, throughput: float, packet_loss: float) -> list[str]:
    """Explain user-entered network risk using weighted operational thresholds."""
    causes = []
    latency_threshold = float(df["latency"].quantile(0.75))
    throughput_threshold = float(df["throughput"].quantile(0.25))

    if packet_loss > 0.5:
        causes.append("High packet loss → unstable transmission")
    if latency > latency_threshold:
        causes.append("High latency → routing delay")
    if throughput < throughput_threshold:
        causes.append("Low throughput → bandwidth bottleneck")

    if not causes:
        causes.append("No major issue detected")

    return causes


def show_ml_pipeline_overview() -> None:
    """Render the ML Pipeline flow with 7 stages as interactive cards."""
    st.subheader("🔄 ML Pipeline Overview")

    pipeline_stages = [
        {
            "icon": "📊",
            "title": "Data Collection",
            "description": "Raw CICIDS traffic ingestion",
            "highlight": False,
        },
        {
            "icon": "🔧",
            "title": "Data Processing",
            "description": "Cleaning & normalization",
            "highlight": False,
        },
        {
            "icon": "⚙️",
            "title": "Feature Engineering",
            "description": "Latency, throughput, loss",
            "highlight": False,
        },
        {
            "icon": "🤖",
            "title": "Model Training",
            "description": "Random Forest classifier",
            "highlight": False,
        },
        {
            "icon": "🎯",
            "title": "Prediction",
            "description": "Congestion detection",
            "highlight": True,
        },
        {
            "icon": "🧠",
            "title": "Explainability",
            "description": "SHAP interpretability",
            "highlight": False,
        },
        {
            "icon": "💡",
            "title": "Insights & Actions",
            "description": "Recommendations & alerts",
            "highlight": False,
        },
    ]

    cols = st.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])

    for idx, stage in enumerate(pipeline_stages):
        if idx % 2 == 0:
            col_idx = idx
        else:
            col_idx = idx

        if col_idx < len(cols):
            with cols[col_idx]:
                border_color = "#06b6d4" if stage["highlight"] else "#475569"
                bg_color = "#164e63" if stage["highlight"] else "#1e293b"
                text_color = "#f8fafc"

                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid {border_color};
                        border-radius: 12px;
                        padding: 16px 12px;
                        background: {bg_color};
                        text-align: center;
                        transition: all 0.3s ease;
                        min-height: 140px;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                    ">
                        <div style="font-size: 2.5rem; margin-bottom: 8px;">{stage['icon']}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; color: {text_color}; margin-bottom: 4px;">
                            {stage['title']}
                        </div>
                        <div style="font-size: 0.8rem; color: #cbd5e1;">
                            {stage['description']}
                        </div>
                        {
                            '<div style="font-size: 0.75rem; color: #06b6d4; margin-top: 8px; font-weight: 600;">ACTIVE</div>'
                            if stage['highlight']
                            else ''
                        }
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            with cols[col_idx]:
                st.markdown(
                    """
                    <div style="
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 140px;
                        font-size: 1.8rem;
                        color: #475569;
                    ">
                    →
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("")
    st.info(
        "**Current Stage:** Prediction Engine Active | The model analyzes network KPIs in real-time "
        "to detect congestion and provide actionable insights."
    )


def overview_tab(df: pd.DataFrame, df_sample: pd.DataFrame, model) -> None:
    """Render high-level network KPIs."""
    st.header("📊 Overview")
    st.write("")

    metrics = compute_dashboard_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Records", f"{metrics['rows']:,}", "#2563eb")
    with col2:
        kpi_card(
            "Congestion Rate",
            f"{metrics['congestion_rate']:.2f}%",
            "#dc2626" if metrics["congestion_rate"] > 70 else "#f59e0b",
        )
    with col3:
        kpi_card("Avg Latency", f"{metrics['avg_latency']:,.2f}", "#7c3aed")
    with col4:
        kpi_card("Avg Throughput", f"{metrics['avg_throughput']:,.2f}", "#059669")

    st.markdown("---")
    show_ml_pipeline_overview()

    st.markdown("---")
    st.subheader("Model Performance")
    try:
        model_metrics = compute_model_metrics(model, df)
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        metric_col1.metric("Accuracy", f"{model_metrics['accuracy'] * 100:.2f}%")
        metric_col2.metric("Precision", f"{model_metrics['precision'] * 100:.2f}%")
        metric_col3.metric("Recall", f"{model_metrics['recall'] * 100:.2f}%")
        metric_col4.metric("F1 Score", f"{model_metrics['f1'] * 100:.2f}%")
    except Exception as e:
        st.warning(f"Unable to compute model metrics: {e}")

    st.markdown("---")

    summary_col, preview_col = st.columns([1, 2])
    with summary_col:
        st.subheader("Dataset Summary")
        st.metric("Rows", f"{metrics['rows']:,}")
        st.metric("Features", metrics["features"])
        st.metric("Missing Values", metrics["missing_values"])

    with preview_col:
        st.subheader("Data Preview")
        rows = st.selectbox("Select rows to display", [50, 100, 500, "All"])
        if rows == "All":
            st.warning("Large dataset may slow UI. Showing a 1,000-row sample.")
            display_df = df.sample(min(len(df), 1000), random_state=42)
        else:
            display_df = df.head(int(rows))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Latency Distribution")
    try:
        hist_df = prepare_visual_df(df_sample)
        fig, ax = plt.subplots(figsize=(6, 2.6), facecolor="#0f172a")
        ax.hist(hist_df["log_latency"].values, bins=40, color="#38bdf8", alpha=0.92)
        style_dark_axes(ax, xlabel="Log Latency", ylabel="Records")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Plot error: {e}")


def network_analytics_tab(
    df: pd.DataFrame,
    df_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model,
    shap_sample: pd.DataFrame,
) -> None:
    """Render latency, throughput, and correlation analytics."""
    st.header("📈 Network Analytics")

    plot_df = prepare_visual_df(df_sample)
    plot_df = plot_df.assign(congestion_label=plot_df["congestion"].map({0: "Normal", 1: "Congested"}))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latency vs Throughput Density")
        try:
            density_df = clean_plot_df(plot_df, ["log_latency", "log_throughput"])
            fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#0f172a")
            image = ax.hexbin(
                density_df["log_latency"].values,
                density_df["log_throughput"].values,
                gridsize=40,
                mincnt=1,
                cmap="turbo",
            )
            style_dark_axes(ax, xlabel="Log Latency", ylabel="Log Throughput")
            colorbar = fig.colorbar(image, ax=ax)
            style_colorbar(colorbar, "Flow density")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Plot error: {e}")

    with col2:
        st.subheader("Packet Loss vs Throughput Density")
        try:
            packet_df = clean_plot_df(plot_df, ["packet_loss", "log_throughput"])
            fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#0f172a")
            image = ax.hexbin(
                packet_df["packet_loss"].values,
                packet_df["log_throughput"].values,
                gridsize=40,
                mincnt=1,
                cmap="plasma",
            )
            style_dark_axes(ax, xlabel="Packet Loss", ylabel="Log Throughput")
            colorbar = fig.colorbar(image, ax=ax)
            style_colorbar(colorbar, "Flow density")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Plot error: {e}")

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Correlation Heatmap")
        try:
            corr_df = clean_plot_df(plot_df, FEATURE_COLUMNS)
            corr = corr_df[FEATURE_COLUMNS].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0f172a")
            heatmap = sns.heatmap(
                corr,
                annot=True,
                cmap="mako",
                center=0,
                fmt=".2f",
                linewidths=0.5,
                linecolor="#334155",
                cbar_kws={"label": "Correlation"},
                ax=ax,
            )
            style_dark_axes(ax)
            for text in ax.texts:
                text.set_color("#f8fafc")
            if heatmap.collections and heatmap.collections[0].colorbar:
                style_colorbar(heatmap.collections[0].colorbar, "Correlation")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Plot error: {e}")

    with col4:
        st.subheader("Latency by Congestion State")
        try:
            box_df = clean_plot_df(plot_df, ["log_latency", "congestion_label"])
            fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#0f172a")
            sns.boxplot(
                data=box_df,
                x="congestion_label",
                y="log_latency",
                palette={"Normal": "#22c55e", "Congested": "#f97316"},
                ax=ax,
            )
            style_dark_axes(ax, xlabel="Congestion", ylabel="Log Latency")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Plot error: {e}")

    st.divider()
    st.subheader("Throughput Shape by Congestion")
    try:
        violin_df = clean_plot_df(plot_df, ["log_throughput", "congestion_label"])
        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0f172a")
        sns.violinplot(
            data=violin_df,
            x="congestion_label",
            y="log_throughput",
            palette={"Normal": "#22c55e", "Congested": "#f97316"},
            cut=0,
            ax=ax,
        )
        style_dark_axes(ax, xlabel="Congestion", ylabel="Log Throughput")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Plot error: {e}")

    st.markdown("---")
    st.subheader("Relationship Interpretation")
    try:
        corr_df = clean_plot_df(plot_df, FEATURE_COLUMNS)
        corr = corr_df[FEATURE_COLUMNS].corr(numeric_only=True)
        packet_throughput = corr.loc["packet_loss", "throughput"]
        latency_throughput = corr.loc["latency", "throughput"]

        if packet_throughput < -0.30:
            st.warning("Higher packet loss strongly reduces throughput.")
        elif packet_throughput > 0.30:
            st.info("Packet loss rises with throughput pressure, suggesting overloaded traffic windows.")
        else:
            st.info("Packet loss has a weak direct relationship with throughput in the sampled data.")

        if latency_throughput < -0.30:
            st.warning("Higher latency is associated with lower throughput.")
        elif latency_throughput > 0.30:
            st.info("Latency and throughput rise together, indicating heavier flows may be stressing the network.")
        else:
            st.info("Latency and throughput show a weak linear relationship, so congestion is likely multi-factor.")
    except Exception as e:
        st.error(f"Plot error: {e}")

    st.divider()
    root_cause_panel(feature_importance)

    st.markdown("---")
    st.subheader("SHAP Explainability Summary")
    show_shap_summary(model, shap_sample)


def time_intelligence_tab(df: pd.DataFrame) -> None:
    """Render rolling congestion trend and spike warnings."""
    st.header("⏱️ Time Intelligence")

    chart_df = df.head(min(len(df), VIZ_SAMPLE_SIZE))
    time_df = clean_plot_df(chart_df[["time", "congestion", "latency"]], ["time", "congestion", "latency"]).copy()
    if time_df.empty:
        st.warning("No data available")
        return

    series = time_df["congestion"].values
    congestion = pd.Series(series).rolling(window=50, min_periods=1).mean() * 100
    rolling_mean = congestion.rolling(window=50, min_periods=1).mean()
    latency_rolling = time_df["latency"].rolling(50, min_periods=1).mean()
    spike_threshold = congestion.quantile(0.90)
    spikes = congestion > spike_threshold

    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_df["time"].values, congestion.values, color="#6baed6", alpha=0.45, label="Congestion")
        ax.plot(time_df["time"].values, rolling_mean.values, color="#08519c", linewidth=2, label="Rolling mean")
        ax.scatter(time_df.loc[spikes.values, "time"].values, congestion.loc[spikes].values, color="#de2d26", s=12, label="Spike")
        ax.axhline(spike_threshold, color="#fdae6b", linestyle="--", linewidth=1, label="90th percentile")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Congestion (%)")
        ax.legend()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Plot error: {e}")

    st.subheader("Latency Rolling Average")
    latency_chart = pd.DataFrame(
        {
            "Latency": time_df["latency"].values,
            "Latency Rolling Avg (50)": latency_rolling.values,
        },
        index=time_df["time"].values,
    )
    st.line_chart(latency_chart)

    recent_spike = bool(spikes.tail(50).any())
    latest_congestion = float(congestion.iloc[-1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest rolling congestion", f"{latest_congestion:.2f}%")
    col2.metric("Spike threshold", f"{spike_threshold:.2f}%")
    col3.metric("Recent spikes", int(spikes.tail(50).sum()))

    if recent_spike:
        st.warning("⚠️ High congestion detected in recent period")
    else:
        st.success("Recent period is below the high-congestion spike threshold")


def root_cause_panel(feature_importance: pd.DataFrame) -> None:
    """Render top model-driven root causes."""
    st.subheader("🔎 Root Cause Analysis")

    if feature_importance.empty:
        st.info("The loaded model does not expose feature_importances_.")
        return

    top_features = feature_importance.head(3).copy()
    causes = [readable_feature_name(feature) for feature in top_features["feature"]]
    st.info("Top model drivers contributing to congestion:")
    for index, cause in enumerate(causes, start=1):
        st.write(f"{index}. {cause}")

    try:
        plot_df = clean_plot_df(top_features, ["importance", "feature"])
        st.subheader("Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(data=plot_df, x="importance", y="feature", color="#756bb1", ax=ax)
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as e:
        st.error(f"Plot error: {e}")


def tower_optimization_tab(df: pd.DataFrame) -> str:
    """Render tower congestion optimization view."""
    st.header("🗼 Tower Optimization")

    tower_summary = (
        df.groupby("tower", observed=False)
        .agg(records=("congestion", "size"), congestion_rate=("congestion", "mean"))
        .reset_index()
    )
    tower_summary["congestion_percentage"] = tower_summary["congestion_rate"] * 100
    tower_summary = tower_summary.sort_values("congestion_percentage", ascending=False).reset_index(drop=True)
    tower_summary["rank"] = tower_summary.index + 1

    most_congested = tower_summary.iloc[0]["tower"]
    worst_congestion = float(tower_summary.iloc[0]["congestion_percentage"])

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            plot_df = clean_plot_df(tower_summary, ["tower", "congestion_percentage"])
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=plot_df, x="tower", y="congestion_percentage", color="#3182bd", ax=ax)
            ax.set_xlabel("Tower")
            ax.set_ylabel("Congestion (%)")
            ax.set_ylim(0, max(100, plot_df["congestion_percentage"].max() * 1.15))
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Plot error: {e}")

    with col2:
        st.metric("Most congested tower", f"Tower {most_congested}")
        if worst_congestion > 90:
            st.error(f"Tower {most_congested} is critically congested")
        elif worst_congestion > 70:
            st.warning(f"Tower {most_congested} needs monitoring")
        else:
            st.success(f"Tower {most_congested} is the highest risk tower, but below critical thresholds")
        st.dataframe(
            tower_summary[["rank", "tower", "records", "congestion_percentage"]],
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Tower Action Plan")
    for _, row in tower_summary.iterrows():
        tower = row["tower"]
        congestion = float(row["congestion_percentage"])

        if congestion > 90:
            status = "Critical congestion"
            action = "Upgrade tower capacity and add a new node"
            st.error(f"Tower {tower} -> {status} -> {action}")
        elif congestion > 70:
            status = "Moderate congestion"
            action = "Monitor usage and prepare load balancing"
            st.warning(f"Tower {tower} -> {status} -> {action}")
        else:
            status = "Stable load"
            action = "Monitor usage"
            st.success(f"Tower {tower} -> {status} -> {action}")

    return str(most_congested)


def prediction_control_tab(df: pd.DataFrame, predictor: Predictor, feature_importance: pd.DataFrame) -> None:
    """Render smart prediction and control panel."""
    st.header("🚀 Prediction & Control")

    st.sidebar.header("🚀 Smart Prediction Panel")

    latency = st.sidebar.slider("Latency", 0, 5000, 100)
    packet_loss = st.sidebar.slider("Packet Loss", 0.0, 1.0, 0.01)
    throughput = st.sidebar.slider("Throughput", 0.0, 1.0, 0.5)

    st.subheader("Input Controls")
    input_col1, input_col2, input_col3 = st.columns(3)
    input_col1.metric("Latency", latency)
    input_col2.metric("Packet Loss", f"{packet_loss:.2f}")
    input_col3.metric("Throughput", f"{throughput:.2f}")

    if packet_loss > 0.2:
        warning = "High Packet Loss"
    elif latency > 1000:
        warning = "High Latency"
    elif throughput < 0.2:
        warning = "Low Throughput"
    else:
        warning = "System Normal"

    st.divider()
    st.caption("Prediction uses the trained model loaded from models/network_model.pkl.")

    if st.sidebar.button("🚀 Analyze Network", type="primary"):
        pred, prob = predictor.predict(latency, throughput, packet_loss)
        prediction = "Congestion Detected" if int(pred) == 1 else "Normal Network"
        input_row = pd.DataFrame(
            [{"latency": latency, "throughput": throughput, "packet_loss": packet_loss}],
            columns=FEATURE_COLUMNS,
        )

        st.subheader("Prediction")
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Network State", prediction)
        result_col2.metric("Congestion Probability", f"{prob:.1%}")
        result_col3.metric("Alert Level", alert_from_probability(prob))

        show_alert(prob)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Warning")
            if warning == "System Normal":
                st.success(warning)
            else:
                st.warning(warning)

            st.subheader("Root Cause Analysis")
            for cause in input_root_causes(df, latency, throughput, packet_loss):
                st.write(f"• {cause}")

        with col2:
            st.subheader("Suggested Actions")
            for suggestion in dynamic_suggestions(df, latency, throughput, packet_loss, prob):
                st.markdown(f"- {suggestion}")

        st.markdown("---")
        st.subheader("SHAP Local Explanation")
        show_shap_force_plot(predictor.model, input_row)
    else:
        st.info("Set the sidebar inputs and run analysis to score the current network state.")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="AI Network Performance Optimization",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 AI Network Performance Optimization")
    st.caption("Real CICIDS data | Cached ML pipeline features | Trained congestion model")

    if not os.path.exists(MODEL_PATH):
        st.error("Model not found. Please run training first.")
        st.stop()

    try:
        predictor = load_predictor()
    except Exception as error:
        st.error(f"Unable to load trained model from {MODEL_PATH}: {error}")
        st.stop()

    try:
        df = load_dashboard_data()
    except Exception as error:
        st.error(f"Unable to load real data from data/raw: {error}")
        st.stop()

    if df.empty:
        st.warning("No data available")
        st.stop()

    df_sample = sample_for_visualization(df)
    shap_sample = sample_for_visualization(df, SHAP_SAMPLE_SIZE)[FEATURE_COLUMNS]
    feature_importance = get_model_feature_importance(predictor)

    tabs = st.tabs(
        [
            "Overview",
            "Network Analytics",
            "Time Intelligence",
            "Tower Optimization",
            "Prediction & Control",
        ]
    )

    with tabs[0]:
        overview_tab(df, df_sample, predictor.model)

    with tabs[1]:
        network_analytics_tab(df, df_sample, feature_importance, predictor.model, shap_sample)

    with tabs[2]:
        time_intelligence_tab(df)

    with tabs[3]:
        tower_optimization_tab(df)

    with tabs[4]:
        prediction_control_tab(df, predictor, feature_importance)


if __name__ == "__main__":
    main()
