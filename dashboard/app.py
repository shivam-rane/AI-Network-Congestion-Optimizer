"""Professional Streamlit dashboard for AI network performance optimization."""

from __future__ import annotations

import os

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
from src.features import create_features, prepare_model_features
from src.predict import Predictor
from optimizer.analytics import compare_strategies, plot_optimization_trajectory, plot_strategy_comparison
from optimizer.report_generator import generate_optimization_report
try:
    from optimizer.simulation_runner import run_optimization_simulation
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    run_optimization_simulation = None


MODEL_PATH = "models/network_model.pkl"
MAX_RECORDS = 50_000
VIZ_SAMPLE_SIZE = 10_000
SHAP_SAMPLE_SIZE = 500
FEATURE_COLUMNS = ["latency", "throughput", "packet_loss"]


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")


def calc_congestion_prob(latency, packet_loss, throughput_mbps):
    prob = (
        (latency / 5000.0) * 0.4
        + (packet_loss / 1.0) * 0.4
        + (1.0 - min(throughput_mbps / 1000.0, 1.0)) * 0.2
    )
    return float(np.clip(prob, 0.0, 1.0))


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
    metrics_df = clean_plot_df(df, FEATURE_COLUMNS + ["congestion"]).copy()
    metrics_df["latency"] = metrics_df["latency"].clip(upper=metrics_df["latency"].quantile(0.99))
    metrics_df["throughput"] = metrics_df["throughput"].clip(upper=metrics_df["throughput"].quantile(0.99))

    return {
        "rows": len(metrics_df),
        "features": len(FEATURE_COLUMNS),
        "missing_values": int(df[FEATURE_COLUMNS + ["congestion"]].isna().sum().sum()),
        "congestion_rate": float(metrics_df["congestion"].mean() * 100),
        "avg_latency": float(metrics_df["latency"].mean()),
        "avg_throughput": float(metrics_df["throughput"].mean()),
        "latency_threshold": float(metrics_df["latency"].quantile(0.75)),
        "throughput_threshold": float(metrics_df["throughput"].quantile(0.25)),
    }


@st.cache_data
def compute_model_metrics(_model, df: pd.DataFrame) -> dict:
    """Evaluate the loaded model on a fresh 20% split from raw CICIDS data."""
    eval_df = create_features(df)
    eval_df = clean_plot_df(eval_df, FEATURE_COLUMNS + ["congestion"]).copy()

    _, X_test, _, y_test = train_test_split(
        eval_df[FEATURE_COLUMNS],
        eval_df["congestion"].astype(int),
        test_size=0.2,
        random_state=99,
        stratify=eval_df["congestion"].astype(int),
    )

    y_true = y_test.astype(int)
    y_pred = _model.predict(prepare_model_features(X_test))

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
    try:
        model = predictor.model
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1

        importance_model = model
        if not hasattr(importance_model, "feature_importances_") and hasattr(model, "named_steps"):
            importance_model = list(model.named_steps.values())[-1]
            if hasattr(importance_model, "n_jobs"):
                importance_model.n_jobs = 1

        if not hasattr(importance_model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        feature_names = getattr(model, "feature_names_in_", getattr(importance_model, "feature_names_in_", FEATURE_COLUMNS))
        return (
            pd.DataFrame(
                {
                    "feature": list(feature_names),
                    "importance": importance_model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")
        return pd.DataFrame(columns=["feature", "importance"])


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
    is_congested: bool = False,
) -> list[str]:
    """Generate input-specific optimization actions."""
    suggestions = []
    latency_threshold = float(df["latency"].quantile(0.75))
    throughput_threshold = float(df["throughput"].quantile(0.25))
    packet_loss_threshold = float(df["packet_loss"].quantile(0.90))

    if packet_loss > 0.30 or packet_loss > packet_loss_threshold:
        suggestions.append("Check cables, hardware, and reduce packet drops")
    if latency > latency_threshold:
        suggestions.append("Optimize routing paths or reduce hops")
    if throughput < throughput_threshold:
        suggestions.append("Increase bandwidth or load balance traffic")
    if congestion_prob > 0.9:
        suggestions.append("Immediate load balancing required")
    elif is_congested and not suggestions:
        suggestions.append("Investigate model-highlighted congestion and monitor affected flows")

    if not suggestions:
        suggestions.append("System operating normally")

    return suggestions


def input_root_causes(
    df: pd.DataFrame,
    latency: float,
    throughput: float,
    packet_loss: float,
    is_congested: bool = False,
) -> list[str]:
    """Explain user-entered network risk using weighted operational thresholds."""
    causes = []
    latency_threshold = float(df["latency"].quantile(0.75))
    throughput_threshold = float(df["throughput"].quantile(0.25))
    packet_loss_threshold = float(df["packet_loss"].quantile(0.90))

    if packet_loss > 0.30 or packet_loss > packet_loss_threshold:
        causes.append("High packet loss → unstable transmission")
    if latency > latency_threshold:
        causes.append("High latency → routing delay")
    if throughput < throughput_threshold:
        causes.append("Low throughput → bandwidth bottleneck")

    if is_congested and not causes:
        causes.append("Model detected multi-factor congestion risk")
    elif not causes:
        causes.append("No major issue detected")

    return causes


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
    st.subheader("Model Performance")
    try:
        model_metrics = compute_model_metrics(model, get_data())
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

    n_towers = len(tower_summary)
    tower_names = [f"Tower {tower}" for tower in tower_summary["tower"].astype(str)]

    if "tower_loads" not in st.session_state or len(st.session_state["tower_loads"]) != n_towers:
        st.session_state["tower_loads"] = np.random.uniform(30, 95, n_towers)

    if st.button("Analyze Network", type="primary", key="tower_analyze_network"):
        st.session_state["tower_loads"] = np.random.uniform(30, 95, n_towers)

    before_loads = np.asarray(st.session_state["tower_loads"], dtype=float)
    loads = before_loads.copy()
    overloaded = [(i, l) for i, l in enumerate(loads) if l > 75]
    underloaded = [(i, l) for i, l in enumerate(loads) if l < 60]
    new_loads = loads.copy()
    for idx, load in overloaded:
        excess = load - 75
        if underloaded:
            share = excess / len(underloaded)
            for uid, _ in underloaded:
                new_loads[uid] = min(new_loads[uid] + share, 75)
            new_loads[idx] = 75
    after_loads = new_loads

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Overloaded towers (before)",
        int(sum(l > 75 for l in before_loads)),
    )
    col2.metric(
        "Overloaded towers (after)",
        int(sum(l > 75 for l in after_loads)),
    )
    col3.metric(
        "Load variance reduced",
        f"{np.std(before_loads) - np.std(after_loads):.2f}%",
    )

    def tower_load_chart(loads: np.ndarray, title: str):
        """Build the load chart with overload-aware colors."""
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [
            "#ff4d4d" if l > 75 else
            "#fbbf24" if l > 60 else
            "#4ade80" for l in loads
        ]
        bars = ax.bar(tower_names, loads, color=colors)
        ax.axhline(75, color="#ff4d4d", linestyle="--",
                   linewidth=1.5, label="Overload threshold (75%)")
        ax.legend()
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_xlabel("Tower")
        ax.set_ylabel("Load (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        return fig

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = tower_load_chart(before_loads, "Tower Loads Before Redistribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with chart_col2:
        fig = tower_load_chart(after_loads, "Tower Loads After Redistribution")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.divider()
    st.subheader("Observed Tower Congestion")

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            plot_df = clean_plot_df(tower_summary, ["tower", "congestion_percentage"])
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f172a")
            sns.barplot(data=plot_df, x="tower", y="congestion_percentage", color="#3182bd", ax=ax)
            ax.set_xlabel("Tower")
            ax.set_ylabel("Congestion (%)")
            ax.axhline(75, color="#ff4d4d", linestyle="--", linewidth=1.5, label="Overload threshold (75%)")
            ax.legend()
            ax.set_ylim(0, max(100, plot_df["congestion_percentage"].max() * 1.15))
            style_dark_axes(ax, xlabel="Tower", ylabel="Congestion (%)")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.error(f"Plot error: {e}")

    with col2:
        st.metric("Most congested tower", f"Tower {most_congested}")
        if worst_congestion > 75:
            st.error(f"Tower {most_congested} is critically congested")
        elif worst_congestion > 60:
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

        if congestion > 75:
            status = "Critical congestion"
            action = "Upgrade tower capacity and add a new node"
            st.error(f"Tower {tower} -> {status} -> {action}")
        elif congestion > 60:
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

    input_scale_note = (
        "Input scale: latency uses the trained model scale 0-5000; throughput is normalized traffic "
        "pressure 0-1; packet loss is a ratio 0-1."
    )

    def alert_banner(message, level):
        if level == 'Critical':
            bg     = 'rgba(220, 38, 38, 0.18)'
            border = '#ef4444'
            color  = '#fca5a5'
        elif level == 'Warning':
            bg     = 'rgba(202, 138, 4, 0.18)'
            border = '#eab308'
            color  = '#fde047'
        else:
            bg     = 'rgba(34, 197, 94, 0.18)'
            border = '#22c55e'
            color  = '#86efac'

        st.markdown(
            f'<div style="'
            f'background-color:{bg} !important;'
            f'border-left:4px solid {border};'
            f'border-radius:6px;'
            f'padding:14px 18px;'
            f'margin:10px 0;'
            f'color:{color} !important;'
            f'font-size:15px;'
            f'font-weight:600;'
            f'">{message}</div>',
            unsafe_allow_html=True
        )

    def predict_dashboard_inputs(latency_value: float, throughput_value: float, packet_loss_value: float) -> tuple[int, float, pd.DataFrame]:
        model_throughput = 1.0 - float(np.clip(throughput_value, 0.0, 1.0))
        raw_input = pd.DataFrame(
            [{"latency": latency_value, "throughput": model_throughput, "packet_loss": packet_loss_value}],
            columns=FEATURE_COLUMNS,
        )

        expected_features = getattr(predictor.model, "prediction_feature_columns_", FEATURE_COLUMNS)
        if list(raw_input.columns) != list(expected_features):
            raise ValueError("Prediction features do not match training features.")

        model_input = prepare_model_features(raw_input)
        classes = list(getattr(predictor.model, "classes_", []))
        if 1 not in classes:
            raise ValueError("Congestion class label 1 was not found in the trained model.")

        congestion_class_index = classes.index(1)
        congestion_prob = float(predictor.model.predict_proba(model_input)[0][congestion_class_index])
        pred = int(predictor.model.predict(model_input)[0])
        return pred, congestion_prob, model_input

    def optimizer_state(latency_value: float, throughput_value: float, packet_loss_value: float, tower_load_value: float) -> dict:
        """Convert prediction inputs into simulator-friendly units."""
        return {
            "latency": float(latency_value),
            "throughput": float(throughput_value) * 1000.0,
            "packet_loss": float(packet_loss_value),
            "tower_load": float(tower_load_value),
            "congestion_probability": 0.0,
        }

    def simulation_table(simulation_log: list[dict]) -> pd.DataFrame:
        """Build a display table from optimizer simulation logs."""
        display_log = []
        for s in simulation_log:
            display_log.append(
                {
                    "Step": "Start" if s["step"] == 0 else s["step"],
                    "Action": "baseline" if s["step"] == 0 else s["action"],
                    "Latency (units)": round(s["latency"], 1),
                    "Throughput (Mbps)": round(s["throughput"], 1),
                    "Packet Loss": round(s["packet_loss"], 4),
                    "Congestion Prob": round(s["congestion_prob"], 4),
                    "Reward": round(s["reward"], 3),
                }
            )
        return pd.DataFrame(display_log)

    def render_inline_optimization(latency_value: float, throughput_value: float, packet_loss_value: float, tower_load_value: float, congestion_probability: float) -> None:
        """Run and render inline AI optimization results."""
        try:
            st.markdown("---")
            st.subheader("⚡ AI Optimization Results")
            initial_state = {
                "latency": float(latency_value),
                "throughput": float(throughput_value) * 1000.0,
                "packet_loss": float(packet_loss_value),
                "tower_load": float(tower_load_value),
            }
            with st.spinner("Running AI optimization..."):
                optimization_result = run_optimization_simulation(initial_state, n_steps=10)
                optimization_log = optimization_result["simulation_log"]
                summary = optimization_result["summary"]
                comparison = compare_strategies(initial_state)
                report_markdown = generate_optimization_report(optimization_log, comparison, feature_importance.head(5))

            initial = next(s for s in optimization_log if s["step"] == 0)
            final = optimization_log[-1]
            cong_init = calc_congestion_prob(initial["latency"], initial["packet_loss"], initial["throughput"])
            cong_final = calc_congestion_prob(final["latency"], final["packet_loss"], final["throughput"])
            congestion_reduction = (cong_init - cong_final) / max(cong_init, 0.001) * 100
            latency_improvement = (initial["latency"] - final["latency"]) / max(initial["latency"], 1) * 100
            throughput_gain = (final["throughput"] - initial["throughput"]) / max(initial["throughput"], 1) * 100

            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric(
                "Congestion reduced by",
                f"{congestion_reduction:.1f}%",
                delta=f"{congestion_reduction:.1f}%",
                delta_color="normal",
            )
            metric_col2.metric(
                "Latency improved by",
                f"{latency_improvement:.1f}%",
                delta=f"{latency_improvement:.1f}%",
                delta_color="normal",
            )
            metric_col3.metric(
                "Throughput gain",
                f"{throughput_gain:.1f}%",
                delta=f"{throughput_gain:.1f}%",
                delta_color="normal",
            )

            with st.expander("📋 Optimization action sequence"):
                st.write(" → ".join(summary["recommended_action_sequence"]))
                st.dataframe(simulation_table(optimization_log), use_container_width=True, hide_index=True)

            trajectory_fig = plot_optimization_trajectory(optimization_log)
            st.pyplot(trajectory_fig, use_container_width=True)
            plt.close(trajectory_fig)

            st.download_button(
                "📥 Download Optimization Report",
                data=report_markdown,
                file_name="optimization_report.md",
                mime="text/markdown",
            )
        except Exception:
            st.warning("Optimization engine unavailable")

    def render_upload_mode() -> None:
        """Render batch upload prediction and optimization mode."""
        file_uploader = st.file_uploader("Upload CSV", type=["csv"])
        if not st.button("▶ Run Batch Analysis", type="primary"):
            st.info("Upload a CSV with latency, packet_loss, and throughput columns.")
            return

        try:
            if file_uploader is None:
                st.error("Please upload a CSV file before running batch analysis.")
                return

            upload_df = pd.read_csv(file_uploader)
            required_columns = {"latency", "packet_loss", "throughput"}
            missing_columns = sorted(required_columns - set(upload_df.columns))
            if missing_columns:
                st.error("Missing required columns: latency, packet_loss, throughput")
                return

            if len(upload_df) > 500:
                st.info("Processing first 500 rows for performance")
                upload_df = upload_df.head(500).copy()

            upload_df.loc[:, FEATURE_COLUMNS] = prepare_model_features(upload_df[FEATURE_COLUMNS])

            results = []
            with st.spinner("Running batch analysis..."):
                for row_number, (_, row) in enumerate(upload_df.iterrows(), start=1):
                    row_latency = float(row["latency"])
                    row_packet_loss = float(row["packet_loss"])
                    row_throughput = float(row["throughput"])
                    row_tower_load = float(row["tower_load"]) if "tower_load" in upload_df.columns and pd.notna(row["tower_load"]) else 70.0
                    pred, prob, _ = predict_dashboard_inputs(row_latency, row_throughput, row_packet_loss)
                    alert_level = alert_from_probability(prob)
                    network_state = "Congestion Detected" if int(pred) == 1 else "Normal Network"
                    optimization_summary = {
                        "congestion_reduction_pct": 0.0,
                        "latency_improvement_pct": 0.0,
                        "throughput_improvement_pct": 0.0,
                        "recommended_action_sequence": [],
                    }
                    if prob > 0.5 and OPTIMIZER_AVAILABLE and run_optimization_simulation is not None:
                        initial_state = optimizer_state(row_latency, row_throughput, row_packet_loss, row_tower_load)
                        initial_state["congestion_probability"] = float(prob)
                        optimization_summary = run_optimization_simulation(initial_state, n_steps=10)["summary"]

                    results.append(
                        {
                            "Row": row_number,
                            "Latency": row_latency,
                            "Packet_Loss": row_packet_loss,
                            "Throughput": row_throughput,
                            "Congestion_Prob": round(float(prob), 4),
                            "Network_State": network_state,
                            "Alert_Level": alert_level,
                            "Opt_Congestion_Reduction%": round(float(optimization_summary["congestion_reduction_pct"]), 2),
                            "Opt_Latency_Improvement%": round(float(optimization_summary["latency_improvement_pct"]), 2),
                            "Opt_Throughput_Gain%": round(float(optimization_summary["throughput_improvement_pct"]), 2),
                            "Recommended_Action": " → ".join(optimization_summary["recommended_action_sequence"]),
                        }
                    )

            results_df = pd.DataFrame(results)
            congested_mask = results_df["Congestion_Prob"] > 0.5
            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            summary_col1.metric("Total rows analyzed", len(results_df))
            summary_col2.metric(
                "Rows with congestion detected",
                f"{int(congested_mask.sum())} ({congested_mask.mean() * 100:.1f}%)",
            )
            summary_col3.metric("Average congestion probability", f"{results_df['Congestion_Prob'].mean():.1%}")
            avg_improvement = results_df.loc[congested_mask, "Opt_Congestion_Reduction%"].mean() if congested_mask.any() else 0.0
            summary_col4.metric("Avg optimization improvement", f"{avg_improvement:.1f}%")

            def style_alert_row(row):
                alert = str(row.get("Alert_Level", "")).strip()
                if alert == "Critical":
                    bg, fg = "#5c1a1a", "#ffb3b3"
                elif alert == "Warning":
                    bg, fg = "#3d2e00", "#ffe08a"
                else:
                    bg, fg = "#1a3a2a", "#a3d9a5"
                return [f"background-color:{bg}; color:{fg}; font-weight:500;"] * len(row)

            fmt = {
                "Congestion_Prob": "{:.3f}",
                "Opt_Congestion_Reduction%": "{:.1f}",
                "Opt_Latency_Improvement%": "{:.1f}",
                "Opt_Throughput_Gain%": "{:.1f}",
                "Latency": "{:.0f}",
                "Packet_Loss": "{:.4f}",
                "Throughput": "{:.0f}",
            }
            # Only format columns that exist in results_df
            fmt = {k: v for k, v in fmt.items() if k in results_df.columns}

            styled = results_df.style.apply(style_alert_row, axis=1).format(fmt)
            st.dataframe(styled, use_container_width=True)
            st.download_button(
                "📥 Download Batch Report (CSV)",
                data=results_df.to_csv(index=False),
                file_name="batch_optimization_report.csv",
                mime="text/csv",
            )

            congested_rows = results_df[results_df["Congestion_Prob"] > 0.5]

            if len(congested_rows) > 0:
                st.markdown("---")
                st.markdown(f"### ⚡ {len(congested_rows)} congested rows detected")
                st.caption(
                    "Click below to run AI optimization on all congested rows "
                    "and see the predicted post-optimization network state."
                )

                if st.button("🚀 Optimize Congested Rows", type="primary", key="batch_optimize_btn"):
                    st.session_state["run_batch_optimize"] = True

            if (
                st.session_state.get("run_batch_optimize", False)
                and len(congested_rows) > 0
                and (not OPTIMIZER_AVAILABLE or run_optimization_simulation is None)
            ):
                st.warning(
                    "Optimizer module not found. "
                    "Make sure optimizer/simulation_runner.py exists."
                )

            if (
                st.session_state.get("run_batch_optimize", False)
                and len(congested_rows) > 0
                and OPTIMIZER_AVAILABLE
                and run_optimization_simulation is not None
            ):
                with st.spinner("🔄 Running AI optimization on congested rows..."):

                    opt_results = []

                    for _, row in congested_rows.iterrows():
                        try:
                            initial_state = {
                                "latency": float(row["Latency"]),
                                "throughput": float(row["Throughput"]) * 1000.0,
                                "packet_loss": float(row["Packet_Loss"]),
                                "tower_load": float(row.get("tower_load", 60.0)),
                            }
                            simulation_result = run_optimization_simulation(initial_state, n_steps=10)
                            sim_log = (
                                simulation_result.get("simulation_log", [])
                                if isinstance(simulation_result, dict)
                                else simulation_result
                            )
                            final = sim_log[-1]
                            initial = sim_log[0]

                            # Recalculate post-optimization congestion
                            post_lat = final["latency"]
                            post_tp = final["throughput"]
                            post_pl = final["packet_loss"]
                            post_cong = round(calc_congestion_prob(post_lat, post_pl, post_tp), 3)

                            if post_cong > 0.7:
                                post_state = "Congestion Detected"
                                post_alert = "Critical"
                            elif post_cong > 0.4:
                                post_state = "Warning"
                                post_alert = "Warning"
                            else:
                                post_state = "Normal Network"
                                post_alert = "Normal"

                            actions = " → ".join([s["action"] for s in sim_log if s["step"] > 0])

                            opt_results.append(
                                {
                                    "Row": int(row["Row"]),
                                    "Before_Latency": round(initial["latency"], 1),
                                    "Before_PktLoss": round(initial["packet_loss"], 4),
                                    "Before_Throughput": round(initial["throughput"], 1),
                                    "Before_Congestion": round(float(row["Congestion_Prob"]), 3),
                                    "After_Latency": round(post_lat, 1),
                                    "After_PktLoss": round(post_pl, 4),
                                    "After_Throughput": round(post_tp, 1),
                                    "After_Congestion": post_cong,
                                    "After_State": post_state,
                                    "After_Alert": post_alert,
                                    "Congestion_Cut_By": round(
                                        (float(row["Congestion_Prob"]) - post_cong)
                                        / float(row["Congestion_Prob"])
                                        * 100,
                                        1,
                                    ),
                                    "Action_Sequence": actions,
                                }
                            )
                        except Exception as e:
                            opt_results.append(
                                {
                                    "Row": int(row["Row"]),
                                    "After_State": f"Error: {e}",
                                    "After_Alert": "Normal",
                                }
                            )

                    opt_df = __import__("pandas").DataFrame(opt_results)
                    if "Congestion_Cut_By" not in opt_df.columns:
                        opt_df["Congestion_Cut_By"] = 0.0

                    # ── Summary metrics ──────────────────────────────
                    st.markdown("### ✅ Optimization Results")
                    c1, c2, c3, c4 = st.columns(4)
                    resolved = opt_df[opt_df["After_Alert"] == "Normal"]
                    c1.metric("Rows optimized", len(opt_df))
                    c2.metric("Now Normal Network", len(resolved), delta=f"{len(resolved) / len(opt_df) * 100:.0f}%")
                    c3.metric("Avg congestion cut", f"{opt_df['Congestion_Cut_By'].mean():.1f}%")
                    c4.metric("Still congested", len(opt_df) - len(resolved), delta_color="inverse")

                    # ── Color-coded before/after table ───────────────
                    def style_opt_row(row):
                        alert = str(row.get("After_Alert", "")).strip()
                        if alert == "Critical":
                            bg, fg = "#5c1a1a", "#ffb3b3"
                        elif alert == "Warning":
                            bg, fg = "#3d2e00", "#ffe08a"
                        else:
                            bg, fg = "#1a3a2a", "#a3d9a5"
                        return [f"background-color:{bg}; color:{fg}; " f"font-weight:500;"] * len(row)

                    cols_to_show = [
                        c
                        for c in [
                            "Row",
                            "Before_Congestion",
                            "After_Congestion",
                            "Congestion_Cut_By",
                            "After_State",
                            "After_Alert",
                            "After_Latency",
                            "After_PktLoss",
                            "After_Throughput",
                            "Action_Sequence",
                        ]
                        if c in opt_df.columns
                    ]

                    fmt_opt = {
                        "Before_Congestion": "{:.3f}",
                        "After_Congestion": "{:.3f}",
                        "Congestion_Cut_By": "{:.1f}%",
                        "After_Latency": "{:.1f}",
                        "After_PktLoss": "{:.4f}",
                        "After_Throughput": "{:.1f}",
                    }
                    fmt_opt = {k: v for k, v in fmt_opt.items() if k in opt_df.columns}

                    styled_opt = opt_df[cols_to_show].style.apply(style_opt_row, axis=1).format(fmt_opt)

                    st.dataframe(styled_opt, use_container_width=True)

                    # ── Download optimized results ────────────────────
                    csv_opt = opt_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "📥 Download Optimized Results (CSV)",
                        data=csv_opt,
                        file_name="optimized_network_results.csv",
                        mime="text/csv",
                        key="dl_opt_results",
                    )

                    # ── Reset button ──────────────────────────────────
                    if st.button("🔁 Reset Optimization", key="reset_opt"):
                        st.session_state["run_batch_optimize"] = False
                        st.rerun()
        except Exception as error:
            st.error(f"Batch analysis failed: {error}")

    def render_upload_mode() -> None:
        """Render batch upload prediction and optimization mode with persisted results."""
        if "batch_results_df" not in st.session_state:
            st.session_state["batch_results_df"] = None
        if "batch_congested_rows" not in st.session_state:
            st.session_state["batch_congested_rows"] = None
        if "run_batch_optimize" not in st.session_state:
            st.session_state["run_batch_optimize"] = False
        if "opt_results_df" not in st.session_state:
            st.session_state["opt_results_df"] = None

        file_uploader = st.file_uploader("Upload CSV", type=["csv"])
        run_batch_analysis = st.button("▶ Run Batch Analysis", type="primary")

        if run_batch_analysis:
            try:
                if file_uploader is None:
                    st.error("Please upload a CSV file before running batch analysis.")
                    return

                upload_df = pd.read_csv(file_uploader)
                required_columns = {"latency", "packet_loss", "throughput"}
                missing_columns = sorted(required_columns - set(upload_df.columns))
                if missing_columns:
                    st.error("Missing required columns: latency, packet_loss, throughput")
                    return

                if len(upload_df) > 500:
                    st.info("Processing first 500 rows for performance")
                    upload_df = upload_df.head(500).copy()

                upload_df.loc[:, FEATURE_COLUMNS] = prepare_model_features(upload_df[FEATURE_COLUMNS])

                results = []
                with st.spinner("Running batch analysis..."):
                    for row_number, (_, row) in enumerate(upload_df.iterrows(), start=1):
                        row_latency = float(row["latency"])
                        row_packet_loss = float(row["packet_loss"])
                        row_throughput = float(row["throughput"])
                        row_tower_load = float(row["tower_load"]) if "tower_load" in upload_df.columns and pd.notna(row["tower_load"]) else 70.0
                        pred, prob, _ = predict_dashboard_inputs(row_latency, row_throughput, row_packet_loss)
                        alert_level = alert_from_probability(prob)
                        network_state = "Congestion Detected" if int(pred) == 1 else "Normal Network"
                        optimization_summary = {
                            "congestion_reduction_pct": 0.0,
                            "latency_improvement_pct": 0.0,
                            "throughput_improvement_pct": 0.0,
                            "recommended_action_sequence": [],
                        }
                        if prob > 0.5 and OPTIMIZER_AVAILABLE and run_optimization_simulation is not None:
                            initial_state = optimizer_state(row_latency, row_throughput, row_packet_loss, row_tower_load)
                            initial_state["congestion_probability"] = float(prob)
                            optimization_summary = run_optimization_simulation(initial_state, n_steps=10)["summary"]

                        results.append(
                            {
                                "Row": row_number,
                                "Latency": row_latency,
                                "Packet_Loss": row_packet_loss,
                                "Throughput": row_throughput,
                                "Congestion_Prob": round(float(prob), 4),
                                "Network_State": network_state,
                                "Alert_Level": alert_level,
                                "Opt_Congestion_Reduction%": round(float(optimization_summary["congestion_reduction_pct"]), 2),
                                "Opt_Latency_Improvement%": round(float(optimization_summary["latency_improvement_pct"]), 2),
                                "Opt_Throughput_Gain%": round(float(optimization_summary["throughput_improvement_pct"]), 2),
                                "Recommended_Action": " → ".join(optimization_summary["recommended_action_sequence"]),
                            }
                        )

                results_df = pd.DataFrame(results)
                st.session_state["batch_results_df"] = results_df
                st.session_state["batch_congested_rows"] = results_df[results_df["Congestion_Prob"] > 0.5]
                st.session_state["run_batch_optimize"] = False
                st.session_state["opt_results_df"] = None
            except Exception as error:
                st.error(f"Batch analysis failed: {error}")

        if st.session_state["batch_results_df"] is None:
            st.info("Upload a CSV with latency, packet_loss, and throughput columns.")
            return

        results_df = st.session_state["batch_results_df"]
        congested_rows = st.session_state["batch_congested_rows"]
        if congested_rows is None:
            congested_rows = results_df[results_df["Congestion_Prob"] > 0.5]
            st.session_state["batch_congested_rows"] = congested_rows

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total rows analyzed", len(results_df))
        n_cong = len(congested_rows)
        col2.metric("Rows with congestion detected", f"{n_cong} ({n_cong / len(results_df) * 100:.1f}%)")
        col3.metric("Average congestion probability", f"{results_df['Congestion_Prob'].mean() * 100:.1f}%")
        if "Opt_Congestion_Reduction%" in results_df.columns:
            col4.metric("Avg optimization improvement", f"{results_df['Opt_Congestion_Reduction%'].mean():.1f}%")

        def style_alert_row(row):
            alert = str(row.get("Alert_Level", "")).strip()
            if alert == "Critical":
                bg, fg = "#5c1a1a", "#ffb3b3"
            elif alert == "Warning":
                bg, fg = "#3d2e00", "#ffe08a"
            else:
                bg, fg = "#1a3a2a", "#a3d9a5"
            return [f"background-color:{bg}; color:{fg}; font-weight:500;"] * len(row)

        fmt = {
            "Congestion_Prob": "{:.3f}",
            "Opt_Congestion_Reduction%": "{:.1f}",
            "Opt_Latency_Improvement%": "{:.1f}",
            "Opt_Throughput_Gain%": "{:.1f}",
            "Latency": "{:.0f}",
            "Packet_Loss": "{:.4f}",
            "Throughput": "{:.0f}",
        }
        fmt = {k: v for k, v in fmt.items() if k in results_df.columns}

        styled = results_df.style.apply(style_alert_row, axis=1).format(fmt)
        st.dataframe(styled, use_container_width=True)
        st.download_button(
            "📥 Download Batch Report (CSV)",
            data=results_df.to_csv(index=False),
            file_name="batch_optimization_report.csv",
            mime="text/csv",
        )

        if n_cong > 0:
            st.markdown("---")
            st.markdown(f"### ⚡ {n_cong} congested rows detected")
            st.caption(
                "Click below to run AI optimization on all congested rows "
                "and see the predicted post-optimization network state."
            )
            if st.button("🚀 Optimize Congested Rows", type="primary", key="batch_optimize_btn"):
                st.session_state["run_batch_optimize"] = True

        if st.session_state.get("run_batch_optimize", False):
            if not OPTIMIZER_AVAILABLE or run_optimization_simulation is None:
                st.warning("Optimizer module not found. Make sure optimizer/simulation_runner.py exists.")
            elif st.session_state["opt_results_df"] is None:
                with st.spinner("🔄 Running AI optimization..."):
                    opt_results = []
                    for _, row in congested_rows.iterrows():
                        try:
                            initial_state = {
                                "latency": float(row["Latency"]),
                                "throughput": float(row["Throughput"]) * 1000.0,
                                "packet_loss": float(row["Packet_Loss"]),
                                "tower_load": float(row.get("tower_load", 60.0)),
                            }
                            simulation_result = run_optimization_simulation(initial_state, n_steps=10)
                            sim_log = (
                                simulation_result.get("simulation_log", [])
                                if isinstance(simulation_result, dict)
                                else simulation_result
                            )
                            final = sim_log[-1]
                            initial = sim_log[0]

                            post_lat = final["latency"]
                            post_tp = final["throughput"]
                            post_pl = final["packet_loss"]
                            post_cong = round(calc_congestion_prob(post_lat, post_pl, post_tp), 3)

                            if post_cong > 0.7:
                                post_state = "Congestion Detected"
                                post_alert = "Critical"
                            elif post_cong > 0.4:
                                post_state = "Warning"
                                post_alert = "Warning"
                            else:
                                post_state = "Normal Network"
                                post_alert = "Normal"

                            actions = " → ".join([s["action"] for s in sim_log if s["step"] > 0])

                            opt_results.append(
                                {
                                    "Row": int(row["Row"]),
                                    "Before_Latency": round(initial["latency"], 1),
                                    "Before_PktLoss": round(initial["packet_loss"], 4),
                                    "Before_Throughput": round(initial["throughput"], 1),
                                    "Before_Congestion": round(float(row["Congestion_Prob"]), 3),
                                    "After_Latency": round(post_lat, 1),
                                    "After_PktLoss": round(post_pl, 4),
                                    "After_Throughput": round(post_tp, 1),
                                    "After_Congestion": post_cong,
                                    "After_State": post_state,
                                    "After_Alert": post_alert,
                                    "Congestion_Cut_By": round(
                                        (float(row["Congestion_Prob"]) - post_cong)
                                        / float(row["Congestion_Prob"])
                                        * 100,
                                        1,
                                    ),
                                    "Action_Sequence": actions,
                                }
                            )
                        except Exception as e:
                            opt_results.append(
                                {
                                    "Row": int(row["Row"]),
                                    "After_State": f"Error: {e}",
                                    "After_Alert": "Normal",
                                }
                            )
                    st.session_state["opt_results_df"] = pd.DataFrame(opt_results)

        if st.session_state["opt_results_df"] is not None:
            opt_df = st.session_state["opt_results_df"]

            st.markdown("### ✅ Optimization Results")
            c1, c2, c3, c4 = st.columns(4)
            resolved = opt_df[opt_df["After_Alert"] == "Normal"]
            c1.metric("Rows optimized", len(opt_df))
            c2.metric("Now Normal Network", len(resolved), delta=f"{len(resolved) / len(opt_df) * 100:.0f}%")
            c3.metric(
                "Avg congestion cut",
                f"{opt_df['Congestion_Cut_By'].mean():.1f}%" if "Congestion_Cut_By" in opt_df.columns else "N/A",
            )
            c4.metric("Still congested", len(opt_df) - len(resolved), delta_color="inverse")

            def style_opt_row(row):
                alert = str(row.get("After_Alert", "")).strip()
                if alert == "Critical":
                    bg, fg = "#5c1a1a", "#ffb3b3"
                elif alert == "Warning":
                    bg, fg = "#3d2e00", "#ffe08a"
                else:
                    bg, fg = "#1a3a2a", "#a3d9a5"
                return [f"background-color:{bg}; color:{fg}; font-weight:500;"] * len(row)

            cols_to_show = [
                c
                for c in [
                    "Row",
                    "Before_Congestion",
                    "After_Congestion",
                    "Congestion_Cut_By",
                    "After_State",
                    "After_Alert",
                    "After_Latency",
                    "After_PktLoss",
                    "After_Throughput",
                    "Action_Sequence",
                ]
                if c in opt_df.columns
            ]

            fmt_opt = {
                "Before_Congestion": "{:.3f}",
                "After_Congestion": "{:.3f}",
                "Congestion_Cut_By": "{:.1f}%",
                "After_Latency": "{:.1f}",
                "After_PktLoss": "{:.4f}",
                "After_Throughput": "{:.1f}",
            }
            fmt_opt = {k: v for k, v in fmt_opt.items() if k in opt_df.columns}

            styled_opt = opt_df[cols_to_show].style.apply(style_opt_row, axis=1).format(fmt_opt)
            st.dataframe(styled_opt, use_container_width=True)

            csv_opt = opt_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Optimized Results (CSV)",
                data=csv_opt,
                file_name="optimized_network_results.csv",
                mime="text/csv",
                key="dl_opt_results",
            )

            if st.button("🔁 Reset Optimization", key="reset_opt"):
                st.session_state["run_batch_optimize"] = False
                st.session_state["opt_results_df"] = None
                st.rerun()

    mode = st.radio("Mode", ["Manual Input", "Upload Dataset"], horizontal=True)
    st.caption(input_scale_note)

    if mode == "Upload Dataset":
        render_upload_mode()
        return

    st.sidebar.header("🚀 Smart Prediction Panel")

    latency = st.sidebar.slider("Latency (model scale 0-5000)", 0, 5000, 100, step=50)
    packet_loss = st.sidebar.slider("Packet Loss (ratio 0-1)", 0.0, 1.0, 0.01, step=0.01)
    throughput = st.sidebar.slider("Throughput (traffic pressure 0-1)", 0.0, 1.0, 0.5, step=0.01)
    tower_load = st.sidebar.slider("Optimization Tower Load (%)", 0.0, 100.0, 70.0)

    st.subheader("Input Controls")
    input_col1, input_col2, input_col3 = st.columns(3)
    input_col1.metric("Latency (model scale)", latency)
    input_col2.metric("Packet Loss ratio", f"{packet_loss:.2f}")
    input_col3.metric("Traffic pressure", f"{throughput:.2f}")

    st.divider()
    st.caption("Prediction uses the trained model loaded from models/network_model.pkl.")

    if st.sidebar.button("🚀 Analyze Network", type="primary"):
        pred, prob, input_row = predict_dashboard_inputs(latency, throughput, packet_loss)
        prediction = "Congestion Detected" if int(pred) == 1 else "Normal Network"
        if prob > 0.70:
            warning = "Critical Congestion Detected"
        elif prob > 0.40:
            warning = "Moderate Congestion Risk"
        else:
            warning = "System Normal"

        st.subheader("Prediction")
        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Network State", prediction)
        result_col2.metric("Congestion Probability", f"{prob:.1%}")
        result_col3.metric("Alert Level", alert_from_probability(prob))

        alert_level = alert_from_probability(prob)
        alert_banner(f"{alert_level} network state | Congestion probability: {prob:.1%}", alert_level)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("System Warning")
            if warning == "System Normal":
                alert_banner(warning, "Normal")
            elif warning == "Moderate Congestion Risk":
                alert_banner(warning, "Warning")
            else:
                alert_banner(warning, "Critical")

            st.subheader("Root Cause Analysis")
            for cause in input_root_causes(df, latency, throughput, packet_loss, int(pred) == 1):
                st.write(f"• {cause}")

        with col2:
            st.subheader("Suggested Actions")
            for suggestion in dynamic_suggestions(df, latency, throughput, packet_loss, prob, int(pred) == 1):
                st.markdown(f"- {suggestion}")

        st.markdown("---")
        st.subheader("SHAP Local Explanation")
        show_shap_force_plot(predictor.model, input_row)

        if prob > 0.5:
            render_inline_optimization(latency, throughput, packet_loss, tower_load, prob)
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
    try:
        feature_importance = get_model_feature_importance(predictor)
    except PermissionError:
        feature_importance = pd.DataFrame(columns=["feature", "importance"])
        st.warning("Feature importance disabled (Windows permission issue)")

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
