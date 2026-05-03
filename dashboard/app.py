"""Telecom Network Intelligence Dashboard."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODEL_FEATURES, PREDICTION_THRESHOLD, TARGET_COLUMN
from src.data_loader import load_cicids_dataset
from src.features import build_features, build_prediction_row
from src.predict import predict_from_features
from src.train import train_model


def alert_level(probability):
    """Map model probability to an operational alert level."""
    if probability < 0.30:
        return "Normal"
    if probability <= 0.70:
        return "Warning"
    return "Critical"


def alert_banner(level):
    """Render colored alert state."""
    if level == "Normal":
        st.success("Normal - network operating within expected range")
    elif level == "Warning":
        st.warning("Warning - congestion risk is increasing")
    else:
        st.error("Critical - congestion likely, immediate optimization recommended")


def dynamic_suggestions(latency, throughput, packet_loss, df, overloaded_tower=None):
    """Generate targeted operational recommendations."""
    suggestions = []

    if latency > df["latency_norm"].quantile(0.75):
        suggestions.append("Reduce routing delay")
    if throughput > df["throughput_norm"].quantile(0.75):
        suggestions.append("Increase bandwidth")
    if packet_loss > df["packet_loss"].quantile(0.75):
        suggestions.append("Improve signal quality")
    if overloaded_tower:
        suggestions.append(f"Reroute traffic away from Tower {overloaded_tower}")

    if not suggestions:
        suggestions.append("Network stable - continue monitoring")

    return suggestions


@st.cache_data(show_spinner=True)
def load_dashboard_dataset():
    """Load CICIDS data and build feature dataframe."""
    raw_df, loaded_rows, original_columns = load_cicids_dataset()
    feature_df = build_features(raw_df)
    return feature_df, loaded_rows, original_columns


@st.cache_resource(show_spinner=True)
def train_dashboard_model(feature_df):
    """Train model once per dashboard session."""
    return train_model(feature_df, tune=False)


def show_kpis(df, metrics, dataset_probabilities):
    """Render top-level KPI cards."""
    congestion_rate = df[TARGET_COLUMN].mean() * 100
    active_alerts = int((dataset_probabilities > 0.70).sum())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Data Points", f"{len(df):,}")
    col2.metric("Congestion Rate", f"{congestion_rate:.2f}%")
    col3.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col4.metric("F1-score", f"{metrics['f1'] * 100:.2f}%")
    col5.metric("Active Alerts", f"{active_alerts:,}")


def show_overview(df, loaded_rows, original_columns, metrics, dataset_probabilities):
    """Overview tab with data quality, metrics, and preview."""
    st.header("Overview")
    show_kpis(df, metrics, dataset_probabilities)
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Loaded CICIDS Rows", f"{loaded_rows:,}")
    col2.metric("Original CICIDS Columns", original_columns)
    col3.metric("Cross-val F1", f"{metrics['cross_val_f1'] * 100:.2f}%")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Validation Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    metric_col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    metric_col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    metric_col4.metric("F1", f"{metrics['f1'] * 100:.2f}%")

    st.write("Confusion matrix:")
    st.write(metrics["confusion_matrix"])

    class_counts = df[TARGET_COLUMN].value_counts().sort_index()
    st.write("Class distribution:")
    st.write(class_counts)

    rows_to_show = st.selectbox("Rows to show", [50, 100, 500, "All"])
    display_df = df if rows_to_show == "All" else df.head(int(rows_to_show))
    st.dataframe(display_df, use_container_width=True)


def show_network_analytics(df, feature_importance):
    """Readable plots with reduced noise."""
    st.header("Network Analytics")

    plot_df = df.sample(min(10_000, len(df)), random_state=42)
    normal = plot_df[plot_df[TARGET_COLUMN] == 0]
    congested = plot_df[plot_df[TARGET_COLUMN] == 1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Latency vs Throughput")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(normal["latency_norm"].to_numpy(), normal["throughput_norm"].to_numpy(), s=10, alpha=0.35, color="green", label="Normal")
        ax.scatter(congested["latency_norm"].to_numpy(), congested["throughput_norm"].to_numpy(), s=10, alpha=0.50, color="red", label="Congested")
        ax.set_xlabel("Normalized Latency")
        ax.set_ylabel("Normalized Throughput")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)

    with col2:
        st.subheader("Anomaly Heatmap")
        corr = df[MODEL_FEATURES + [TARGET_COLUMN]].corr().to_numpy()
        labels = MODEL_FEATURES + [TARGET_COLUMN]
        fig, ax = plt.subplots(figsize=(7, 4))
        image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(feature_importance["feature"].to_numpy(), feature_importance["importance"].to_numpy(), color="#f59e0b")
    ax.set_ylabel("Importance")
    ax.set_xticklabels(feature_importance["feature"].to_numpy(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)


def show_time_intelligence(df):
    """Early warning analysis using rolling congestion trends."""
    st.header("Early Warning System")

    grouped = df.groupby(df["time_index"] // 100)[TARGET_COLUMN].mean()
    rolling = grouped.rolling(window=50, min_periods=1).mean()
    spike_threshold = rolling.quantile(0.90)
    spikes = rolling[rolling > spike_threshold]

    x = grouped.index.to_numpy()
    y = grouped.values
    trend = rolling.values
    spike_x = spikes.index.to_numpy()
    spike_y = spikes.values

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, alpha=0.25, color="gray", label="Congestion")
    ax.plot(x, trend, color="#2563eb", label="Rolling mean")
    ax.scatter(spike_x, spike_y, color="#dc2626", s=24, label="Spike")
    ax.set_xlabel("Time window")
    ax.set_ylabel("Congestion rate")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)

    if len(spikes) > 0:
        st.warning(f"Warning: congestion spikes detected between time windows {int(spikes.index.min())} and {int(spikes.index.max())}.")
        if spikes.max() > 0.70:
            st.error("Critical: rolling congestion crossed 70% in at least one window.")
    else:
        st.success("No major rolling congestion spikes detected.")


def tower_summary(df):
    """Compute tower load and congestion summary."""
    summary = (
        df.groupby("tower")
        .agg(
            traffic_count=("tower", "count"),
            load_percent=("throughput_norm", lambda x: float(x.mean() * 100)),
            congestion_percent=(TARGET_COLUMN, lambda x: float(x.mean() * 100)),
        )
        .reset_index()
        .sort_values("tower")
    )
    return summary


def show_tower_optimization(df):
    """Tower load simulation based on traffic intensity."""
    st.header("Tower Optimization")
    summary = tower_summary(df)
    overloaded = summary.sort_values("congestion_percent", ascending=False).iloc[0]["tower"]
    least_loaded = summary.sort_values("congestion_percent", ascending=True).iloc[0]["tower"]

    st.dataframe(summary, use_container_width=True)
    st.warning(f"Tower {overloaded} is overloaded. Recommended shift: move traffic from Tower {overloaded} to Tower {least_loaded}.")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(summary["tower"].to_numpy(), summary["congestion_percent"].to_numpy(), color=["#2563eb", "#f59e0b", "#16a34a"])
    ax.set_xlabel("Tower")
    ax.set_ylabel("Congestion (%)")
    ax.set_title("Congestion by Tower")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig)

    return overloaded


def show_root_cause(df, feature_importance, latency, throughput, packet_loss):
    """Explain why congestion is likely for the current input."""
    st.subheader("Root Cause Analysis")
    top_features = feature_importance.head(3)["feature"].tolist()
    st.write("Top model drivers:", ", ".join(top_features))

    causes = []
    if latency > df["latency_norm"].quantile(0.75):
        causes.append("high latency")
    if throughput > df["throughput_norm"].quantile(0.75):
        causes.append("high throughput pressure")
    if packet_loss > df["packet_loss"].quantile(0.75):
        causes.append("packet-loss anomaly")

    if causes:
        st.info("Congestion happened because of " + ", ".join(causes) + ".")
    else:
        st.info("No dominant congestion driver is above the current high-risk thresholds.")


def show_prediction_control(df, model, scaler, feature_importance, overloaded_tower):
    """Smart prediction panel."""
    st.header("Smart Prediction Engine")

    st.sidebar.header("Network Input")
    latency = st.sidebar.slider("Normalized Latency", 0.0, 1.0, float(df["latency_norm"].median()))
    throughput = st.sidebar.slider("Normalized Throughput", 0.0, 1.0, float(df["throughput_norm"].median()))
    packet_loss = st.sidebar.slider("Packet Loss", 0.0, 10.0, float(df["packet_loss"].median()))

    input_row = build_prediction_row(latency, throughput, packet_loss)

    if st.button("Analyze Network", type="primary"):
        prediction, probability = predict_from_features(model, scaler, input_row)
        level = alert_level(probability)

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", prediction)
        col2.metric("Congestion Probability", f"{probability * 100:.2f}%")
        col3.metric("Alert Level", level)
        alert_banner(level)

        show_root_cause(df, feature_importance, latency, throughput, packet_loss)

        st.subheader("Dynamic Suggestions")
        for suggestion in dynamic_suggestions(latency, throughput, packet_loss, df, overloaded_tower=overloaded_tower):
            st.write("-", suggestion)


@st.cache_resource(show_spinner=True)
def train_dashboard_model_cached(feature_df):
    """Train and cache model artifacts for the dashboard."""
    return train_model(feature_df, tune=False)


def main():
    """Run the dashboard."""
    st.set_page_config(page_title="AI Network Congestion Optimizer", layout="wide")
    st.title("AI Network Congestion Optimizer")
    st.caption("Production-style CICIDS-based telecom performance optimization")

    try:
        df, loaded_rows, original_columns = load_dashboard_dataset()
    except Exception as error:
        st.error(str(error))
        st.info("Place the real CICIDS CSV files inside data/raw/ and rerun the app.")
        st.stop()

    model, scaler, metrics, feature_importance, dataset_probabilities = train_dashboard_model_cached(df)

    overview_tab, analytics_tab, warning_tab, tower_tab, prediction_tab = st.tabs(
        ["Overview", "Network Analytics", "Early Warning", "Tower Optimization", "Prediction"]
    )

    with overview_tab:
        show_overview(df, loaded_rows, original_columns, metrics, dataset_probabilities)

    with analytics_tab:
        show_network_analytics(df, feature_importance)

    with warning_tab:
        show_time_intelligence(df)

    with tower_tab:
        overloaded_tower = show_tower_optimization(df)

    overloaded_tower = tower_summary(df).sort_values("congestion_percent", ascending=False).iloc[0]["tower"]
    with prediction_tab:
        show_prediction_control(df, model, scaler, feature_importance, overloaded_tower)


if __name__ == "__main__":
    main()
