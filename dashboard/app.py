"""Professional Streamlit dashboard for AI network performance optimization."""

from __future__ import annotations

import os

os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import datetime
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
from src.features import FEATURE_COLUMNS as TRAINING_FEATURE_ORDER, create_features, prepare_model_features
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
TRAIN_ORDER = tuple(TRAINING_FEATURE_ORDER)
FEATURE_COLUMNS = list(TRAIN_ORDER)


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")


def calc_congestion_prob(latency, packet_loss, throughput_mbps):
    throughput_ratio = float(throughput_mbps)
    if throughput_ratio > 1.0:
        throughput_ratio = min(throughput_ratio / 1000.0, 1.0)
    prob = (
        (latency / 5000.0) * 0.4
        + (packet_loss / 1.0) * 0.4
        + (1.0 - np.clip(throughput_ratio, 0.0, 1.0)) * 0.2
    )
    return float(np.clip(prob, 0.0, 1.0))


def style_dark_axes(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Make matplotlib charts readable on Streamlit dark themes."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    style_fig(ax.figure)


def style_colorbar(colorbar, label: str) -> None:
    """Style matplotlib colorbars for dark UI."""
    colorbar.set_label(label, color="#555")
    colorbar.ax.yaxis.set_tick_params(color="#555", labelcolor="#555")
    colorbar.outline.set_edgecolor("#1e1e1e")


def page_title(title, subtitle=""):
    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
      <p style="font-size:0.6rem!important;color:#252525!important;letter-spacing:0.14em;
                margin:0 0 5px;text-transform:uppercase;">{subtitle}</p>
      <h1 style="font-size:1.1rem;font-weight:500;color:#e0e0e0;
                 margin:0;letter-spacing:-0.02em;">{title}</h1>
    </div>""", unsafe_allow_html=True)


def section_label(text):
    st.markdown(
        f'<p style="font-size:0.6rem!important;color:#2a2a2a!important;letter-spacing:0.14em;'
        f'text-transform:uppercase;margin:1.5rem 0 0.6rem;border-top:1px solid #111;'
        f'padding-top:1rem;">'
        f'{text}</p>', unsafe_allow_html=True)


def divider():
    st.markdown('<div style="border-top:1px solid #111;margin:2rem 0;"></div>',
                unsafe_allow_html=True)


def status_card(label, value, color='#e0e0e0'):
    st.markdown(f"""
    <div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:10px;
                padding:1.25rem 1.5rem;">
        <p style="font-size:0.62rem!important;color:#3a3a3a!important;text-transform:uppercase;
                  letter-spacing:0.1em;margin:0 0 10px;font-weight:700;">{label}</p>
        <p style="font-size:1.45rem!important;font-weight:700;color:{color}!important;
                  margin:0;letter-spacing:-0.02em;">{value}</p>
    </div>""", unsafe_allow_html=True)


def alert_banner(message, level):
    colors = {
        'Critical': ('#ef4444', '#1a0505', '#4b1c1c'),
        'Warning': ('#f59e0b', '#1a1205', '#4b3a05'),
        'Normal': ('#16a34a', '#051a0a', '#0a4b1c'),
    }
    text, bg, border = colors.get(level, colors['Normal'])
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="border-left:2px solid {border};background:{bg};
                padding:8px 14px;margin:8px 0;display:flex;
                justify-content:space-between;align-items:center;">
      <span style="font-size:0.72rem;color:{text};letter-spacing:0.04em;">{message}</span>
      <span style="font-size:0.6rem;color:{border};opacity:0.7;">{ts}</span>
    </div>""", unsafe_allow_html=True)


def topbar():
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 0 16px;padding-right:16px;border-bottom:1px solid #111;
                margin-bottom:20px;flex-wrap:nowrap;">
      <span style="font-size:0.62rem;color:#2a2a2a;
                 letter-spacing:0.12em;
                 font-family:'JetBrains Mono',monospace;">
        NETOPT &nbsp;&middot;&nbsp; CICIDS-2017 &nbsp;&middot;&nbsp; RF CLASSIFIER
      </span>
      <div style="display:flex;align-items:center;
                  gap:14px;flex-shrink:0;flex-wrap:nowrap;">
        <span style="font-size:0.58rem;padding:3px 10px;
                    border:1px solid #7f1d1d;color:#ef4444;
                    border-radius:2px;letter-spacing:0.1em;
                    font-family:'JetBrains Mono',monospace;
                    white-space:nowrap;flex-shrink:0;background:transparent;">
          LIVE
        </span>
        <span style="font-size:0.62rem;color:#4a4a4a;
                    font-family:'JetBrains Mono',monospace;
                    white-space:nowrap;min-width:58px;">
          {now}
        </span>
      </div>
    </div>""", unsafe_allow_html=True)


def prediction_metric_grid(network_state, cong_prob, alert_level):
    if cong_prob > 0.70:
        state_color = '#ef4444'
        state_label = 'Congestion Detected'
        alert_label = 'Critical'
    elif cong_prob > 0.40:
        state_color = '#f59e0b'
        state_label = 'Elevated Risk'
        alert_label = 'Warning'
    else:
        state_color = '#22c55e'
        state_label = 'Normal Network'
        alert_label = 'Normal'
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;
                background:#111;border:1px solid #111;border-radius:4px;
                overflow:hidden;margin-bottom:1.5rem;">
      <div style="background:#0a0a0a;padding:14px 16px;">
        <p style="font-size:0.58rem!important;color:#333!important;letter-spacing:0.12em;margin:0 0 8px;">NETWORK STATE</p>
        <p style="font-size:1.3rem!important;font-weight:500;color:{state_color}!important;margin:0;line-height:1;">{state_label}</p>
      </div>
      <div style="background:#0a0a0a;padding:14px 16px;border-left:1px solid #111;">
        <p style="font-size:0.58rem!important;color:#333!important;letter-spacing:0.12em;margin:0 0 8px;">CONGESTION PROB</p>
        <p style="font-size:1.3rem!important;font-weight:500;color:{state_color}!important;margin:0;line-height:1;">{cong_prob*100:.1f}%</p>
      </div>
      <div style="background:#0a0a0a;padding:14px 16px;border-left:1px solid #111;">
        <p style="font-size:0.58rem!important;color:#333!important;letter-spacing:0.12em;margin:0 0 8px;">ALERT LEVEL</p>
        <p style="font-size:1.3rem!important;font-weight:500;color:{state_color}!important;margin:0;line-height:1;">{alert_label}</p>
      </div>
    </div>""", unsafe_allow_html=True)


def render_optimization_log(sim_log, congestion_reduction):
    action_colors = {
        'reduce_load': '#888888',
        'reroute_traffic': '#16a34a',
        'increase_bandwidth': '#a78bfa',
        'rebalance_towers': '#fb923c',
        'do_nothing': '#404040',
        'baseline': '#2a2a2a',
    }

    rows_html = ""
    for s in sim_log:
        action = "baseline" if s['step'] == 0 else s['action']
        ac = action_colors.get(action, '#666')
        step_label = "start" if s['step'] == 0 else f"step {s['step']}"
        cong_color = '#ef4444' if s['congestion_prob'] > 0.7 else '#f59e0b' if s['congestion_prob'] > 0.4 else '#16a34a'
        rows_html += f"""
        <div style="display:flex;gap:12px;padding:4px 0;border-bottom:1px solid #0f0f0f;align-items:baseline;">
          <span style="font-size:0.62rem;color:#333;min-width:40px;">{step_label}</span>
          <span style="font-size:0.7rem;color:{ac};min-width:140px;">{action}</span>
          <span style="font-size:0.65rem;color:#404040;">
            lat <span style="color:#666;">{s['latency']:.0f}</span> ·
            tp <span style="color:#666;">{s['throughput']:.0f}</span> ·
            pl <span style="color:#666;">{s['packet_loss']:.3f}</span> ·
            cong <span style="color:{cong_color};">{s['congestion_prob']:.3f}</span>
          </span>
        </div>"""

    st.markdown(f"""
    <div style="border:1px solid #1a1a1a;border-radius:4px;overflow:hidden;margin-top:1rem;">
      <div style="background:#080808;padding:8px 14px;border-bottom:1px solid #1a1a1a;
                  display:flex;justify-content:space-between;align-items:center;">
        <span style="font-size:0.6rem;color:#333;letter-spacing:0.1em;">
          RL OPTIMIZATION LOG · {len(sim_log)-1}-step rollout
        </span>
        <span style="font-size:0.6rem;color:#16a34a;">
          congestion -{congestion_reduction:.1f}%
        </span>
      </div>
      <div style="padding:8px 14px;background:#0a0a0a;">{rows_html}</div>
    </div>""", unsafe_allow_html=True)


def render_sidebar_navigation() -> str:
    st.sidebar.markdown("""
    <div style="padding:14px 14px 12px;border-bottom:1px solid #111;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:13px;
                   font-weight:500;color:#c0c0c0;letter-spacing:0.04em;">NETOPT</span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
                   color:#2a2a2a;margin-left:6px;">v2.1</span>
    </div>
    """, unsafe_allow_html=True)

    def nav_item(label, icon, page_key, current_page):
        is_active = current_page == page_key
        bg = 'background:#0f0f0f;' if is_active else ''
        border = 'border-left:2px solid #2563eb;' if is_active else 'border-left:2px solid transparent;'
        color = '#d0d0d0' if is_active else '#3a3a3a'
        icon_op = '0.9' if is_active else '0.4'

        st.sidebar.markdown(f"""
        <div style="{bg}{border}display:flex;align-items:center;gap:9px;
                    padding:7px 14px;cursor:pointer;margin:0;">
          <i class="ti {icon}" style="font-size:13px;color:{color};
             opacity:{icon_op};"></i>
          <span style="font-family:'JetBrains Mono',monospace;font-size:12px;
                       color:{color};">{label}</span>
        </div>""", unsafe_allow_html=True)

        clicked = st.sidebar.button(
            label, key=f"navbtn_{page_key}",
            use_container_width=True
        )
        if clicked:
            st.session_state['page'] = page_key
            st.rerun()

    current = st.session_state.get('page', 'overview')

    st.sidebar.markdown("""<div style="padding:4px 0 2px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
      color:#1e1e1e;letter-spacing:0.14em;padding:0 14px;">MONITOR</span>
    </div>""", unsafe_allow_html=True)

    nav_item('Overview', 'ti-layout-dashboard', 'overview', current)
    nav_item('Analytics', 'ti-chart-line', 'analytics', current)
    nav_item('Time intel', 'ti-clock', 'time_intel', current)

    st.sidebar.markdown("""
    <div style="height:1px;background:#111;margin:8px 0;"></div>
    <div style="padding:4px 0 2px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:9px;
      color:#1e1e1e;letter-spacing:0.14em;padding:0 14px;">CONTROL</span>
    </div>
    """, unsafe_allow_html=True)

    nav_item('Towers', 'ti-antenna', 'towers', current)
    nav_item('Predict & control', 'ti-cpu', 'predict', current)

    st.sidebar.markdown("""
    <div style="height:1px;background:#111;margin:10px 0 12px;"></div>
    <div style="margin:0 10px;padding:10px 12px;background:#0a0a0a;
                border:1px solid #161616;border-radius:3px;">
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
                  color:#252525;letter-spacing:0.12em;margin-bottom:5px;">MODEL</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                  color:#666;">RF Classifier</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                  color:#16a34a;">97.11% acc</div>
    </div>
    """, unsafe_allow_html=True)
    return st.session_state['page']


def style_fig(fig):
    fig.patch.set_facecolor('#0e0e0e')
    for ax in fig.get_axes():
        ax.set_facecolor('#0e0e0e')
        ax.tick_params(colors='#333', labelsize=8, length=0)
        ax.xaxis.label.set_color('#333')
        ax.yaxis.label.set_color('#333')
        ax.title.set_color('#252525')
        ax.title.set_fontsize(8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#161616')
        ax.grid(True, color='#141414', linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
    return fig


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
            background: #0d0d0d;
            border: 1px solid #1e1e1e;
            border-radius: 8px;
            padding: 18px 18px 16px 18px;
            min-height: 112px;
        ">
            <div style="font-size: 0.65rem; color: #444; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;">{label}</div>
            <div style="font-size: 1.5rem; color: #e8e8e8; font-weight: 700; margin-top: 10px;">{value}</div>
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
    model = joblib.load(MODEL_PATH)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


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
        alert_banner(f"Feature importance unavailable: {e}", "Warning")
        return pd.DataFrame(columns=["feature", "importance"])


def extract_class_one_shap(raw_shap, congestion_idx: int):
    """Return class-1 SHAP values across SHAP/sklearn output variants."""
    if isinstance(raw_shap, list):
        shap_vals = raw_shap[congestion_idx]
    else:
        raw_shap = np.asarray(raw_shap)
        if len(raw_shap.shape) == 3:
            shap_vals = raw_shap[0, :, congestion_idx] if raw_shap.shape[0] == 1 else raw_shap[:, :, congestion_idx]
        else:
            shap_vals = raw_shap
    return np.asarray(shap_vals)


@st.cache_data(show_spinner="Computing SHAP explanations...")
def compute_shap_values(_model, x_sample: pd.DataFrame):
    """Compute cached SHAP values for a small model-ready sample."""
    if shap is None:
        return None, None, None
    explainer = shap.TreeExplainer(_model)
    feature_order = list(TRAIN_ORDER)
    if isinstance(x_sample, pd.DataFrame):
        sample = x_sample[feature_order].sample(min(len(x_sample), 100), random_state=42)
    else:
        sample = np.asarray(x_sample, dtype=float)
    class_list = list(_model.classes_)
    congestion_idx = class_list.index(1)
    raw_shap = explainer.shap_values(sample)
    shap_values = extract_class_one_shap(raw_shap, congestion_idx)
    return explainer.expected_value, shap_values, sample


def show_feature_importance_fallback(model) -> None:
    """Show feature importance when SHAP cannot render."""
    importance_model = model
    if not hasattr(importance_model, "feature_importances_") and hasattr(model, "named_steps"):
        importance_model = list(model.named_steps.values())[-1]

    if hasattr(importance_model, "feature_importances_"):
        importances = np.asarray(importance_model.feature_importances_, dtype=float)
        features = list(TRAIN_ORDER)
        sorted_idx = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_values = [importances[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#0e0e0e')
        ax.set_facecolor('#0e0e0e')

        ax.barh(sorted_features, sorted_values,
                color='#1e3a5f', edgecolor='none', height=0.32)

        max_val = max(sorted_values)
        for i, (f, v) in enumerate(zip(sorted_features,
                                        sorted_values)):
            if v > max_val * 0.6:
                ax.text(v - max_val*0.02, i,
                        str(round(v,3)),
                        va='center', ha='right',
                        fontsize=7, color='#111',
                        fontfamily='monospace')
            else:
                ax.text(v + max_val*0.02, i,
                        str(round(v,3)),
                        va='center', ha='left',
                        fontsize=7, color='#333',
                        fontfamily='monospace')

        ax.set_xlim(0, max_val * 1.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='#2a2a2a', labelsize=8, length=0)
        ax.set_xlabel('importance score', fontsize=7,
                      color='#222', fontfamily='monospace')
        ax.yaxis.set_tick_params(labelcolor='#444')
        ax.grid(axis='x', color='#161616', linewidth=0.4, alpha=0.6)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.8)
        st.pyplot(style_fig(fig), use_container_width=True)
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

        shap_values_global = shap_values
        mean_shap = np.abs(shap_values_global).mean(axis=0)
        features = list(TRAIN_ORDER)
        sorted_idx = np.argsort(mean_shap)
        sorted_f = [features[i] for i in sorted_idx]
        sorted_s = [mean_shap[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#0e0e0e')
        ax.set_facecolor('#0e0e0e')

        max_s = max(sorted_s) or 1.0
        colors = ['#{:02x}{:02x}{:02x}'.format(
                      int(30 + 160*(v/max_s)),
                      int(58 + 40*(v/max_s)),
                      int(95 + 50*(v/max_s)))
                  for v in sorted_s]

        ax.barh(sorted_f, sorted_s,
                color=colors, edgecolor='none', height=0.32)

        for i, val in enumerate(sorted_s):
            if val > max_s * 0.6:
                ax.text(val - max_s*0.02, i, f'{val:.4f}',
                        va='center', ha='right',
                        fontsize=7, color='#111',
                        fontfamily='monospace')
            else:
                ax.text(val + max_s*0.02, i, f'{val:.4f}',
                        va='center', ha='left',
                        fontsize=7, color='#333',
                        fontfamily='monospace')

        ax.set_xlim(0, max_s * 1.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='#2a2a2a', labelsize=8, length=0)
        ax.set_xlabel('mean |SHAP value|', fontsize=7,
                      color='#222', fontfamily='monospace',
                      labelpad=6)
        ax.yaxis.set_tick_params(labelcolor='#444', labelsize=8)
        ax.set_yticklabels(sorted_f, fontfamily='monospace',
                           fontsize=8, color='#444')
        ax.grid(axis='x', color='#161616',
                linewidth=0.4, alpha=0.6)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.8)
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)
    except Exception as e:
        alert_banner("SHAP not available, showing feature importance instead", "Warning")
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

        class_list = list(model.classes_)
        congestion_idx = class_list.index(1)
        base_value = expected_value[congestion_idx] if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > congestion_idx else expected_value
        local_shap = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        local_sample = sample.iloc[0] if isinstance(sample, pd.DataFrame) else pd.Series(sample[0], index=list(TRAIN_ORDER))
        force_plot = shap.force_plot(base_value, local_shap, local_sample)
        components.html(
            f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>",
            height=260,
            scrolling=True,
        )
    except Exception as e:
        alert_banner("SHAP not available, showing feature importance instead", "Warning")
        st.caption(str(e))
        show_feature_importance_fallback(model)


def render_shap_chart(shap_vals, feat_names):
    import matplotlib.pyplot as plt
    import numpy as np

    sv = list(shap_vals.flatten())
    fn = list(feat_names)

    # Sort by absolute value ascending
    # (largest bar at top in horizontal chart)
    order = sorted(range(len(sv)),
                   key=lambda i: abs(sv[i]))
    sf = [fn[i] for i in order]
    sv = [sv[i] for i in order]

    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor('#0e0e0e')
    ax.set_facecolor('#0e0e0e')

    # Red = positive (pushes to congestion)
    # Blue = negative (reduces congestion)
    colors = ['#ef4444' if v > 0 else '#3b82f6'
              for v in sv]
    ax.barh(sf, sv, color=colors,
            edgecolor='none', height=0.35)
    ax.axvline(0, color='#333', linewidth=0.8)

    # Tight x-axis based on actual values
    max_abs = max(abs(v) for v in sv) if sv else 0.1
    pad     = max_abs * 0.30
    ax.set_xlim(
        min(min(sv), 0) - pad,
        max(max(sv), 0) + pad
    )

    # Non-colliding labels
    for i, v in enumerate(sv):
        label = str(round(v, 3))
        if abs(v) > max_abs * 0.55:
            ax.text(v * 0.88, i, label,
                    va='center',
                    ha='right' if v > 0 else 'left',
                    fontsize=7, color='#111',
                    fontfamily='monospace')
        else:
            off = max_abs * 0.06
            ax.text(v + (off if v >= 0 else -off),
                    i, label,
                    va='center',
                    ha='left' if v >= 0 else 'right',
                    fontsize=7, color='#444',
                    fontfamily='monospace')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors='#2a2a2a',
                   labelsize=8, length=0)
    ax.set_xlabel('SHAP value', fontsize=7,
                  color='#333', fontfamily='monospace')
    ax.yaxis.set_tick_params(
        labelcolor='#555', labelsize=8)
    ax.set_yticklabels(
        sf, fontfamily='monospace',
        fontsize=8, color='#555')
    ax.grid(axis='x', color='#161616',
            linewidth=0.4, alpha=0.6)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_shap_bar_chart(model, input_row: pd.DataFrame) -> None:
    """Render a contained local SHAP bar chart for the prediction panel."""
    try:
        _, shap_values, sample = compute_shap_values(model, input_row)
        if shap_values is None or sample is None:
            show_feature_importance_fallback(model)
            return

        render_shap_chart(shap_values, list(TRAIN_ORDER))
    except Exception as e:
        alert_banner("SHAP not available, showing feature importance instead", "Warning")
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
    if probability > 0.70:
        return "Critical"
    if probability > 0.40:
        return "Warning"
    return "Normal"


def network_state_from_probability(cong_prob: float) -> str:
    if cong_prob > 0.70:
        return 'Congestion Detected'
    if cong_prob > 0.40:
        return 'Elevated Risk'
    return 'Normal Network'


def show_alert(probability: float) -> None:
    """Render probability-based warning state."""
    level = alert_from_probability(probability)
    message = f"{level} network state | Congestion probability: {probability:.1%}"
    alert_banner(message, level)


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
    st.markdown("""
    <p style="font-size:0.6rem;color:#252525;letter-spacing:0.08em;
              margin-bottom:20px;font-family:'JetBrains Mono',monospace;">
      NETOPT / <span style="color:#555;">overview</span>
    </p>""", unsafe_allow_html=True)

    metrics = compute_dashboard_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Records", f"{metrics['rows']:,}", "#888888")
    with col2:
        kpi_card(
            "Congestion Rate",
            f"{metrics['congestion_rate']:.2f}%",
            "#dc2626" if metrics["congestion_rate"] > 70 else "#f59e0b",
        )
    with col3:
        kpi_card("Avg Latency", f"{metrics['avg_latency']:,.2f}", "#7c3aed")
    with col4:
        kpi_card("Avg Throughput", f"{metrics['avg_throughput']:,.2f}", "#16a34a")

    try:
        model_metrics = compute_model_metrics(model, get_data())
        st.markdown(f"""
        <div style="border:1px solid #1a1a1a;border-radius:4px;overflow:hidden;
                    margin-top:24px;">
          <div style="background:#080808;padding:10px 16px;border-bottom:1px solid #1a1a1a;">
            <span style="font-size:0.6rem;color:#252525;letter-spacing:0.14em;
                         font-family:'JetBrains Mono',monospace;">MODEL PERFORMANCE</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1px;
                      background:#111;">
            <div style="background:#0a0a0a;padding:12px 16px;">
              <p style="font-size:0.58rem;color:#333;letter-spacing:0.1em;margin:0 0 6px;
                        font-family:'JetBrains Mono',monospace;">ACCURACY</p>
              <p style="font-size:1.1rem;font-weight:500;color:#e0e0e0;margin:0;
                        font-family:'JetBrains Mono',monospace;">{model_metrics['accuracy'] * 100:.2f}%</p>
            </div>
            <div style="background:#0a0a0a;padding:12px 16px;border-left:1px solid #111;">
              <p style="font-size:0.58rem;color:#333;letter-spacing:0.1em;margin:0 0 6px;
                        font-family:'JetBrains Mono',monospace;">PRECISION</p>
              <p style="font-size:1.1rem;font-weight:500;color:#e0e0e0;margin:0;
                        font-family:'JetBrains Mono',monospace;">{model_metrics['precision'] * 100:.2f}%</p>
            </div>
            <div style="background:#0a0a0a;padding:12px 16px;border-left:1px solid #111;">
              <p style="font-size:0.58rem;color:#333;letter-spacing:0.1em;margin:0 0 6px;
                        font-family:'JetBrains Mono',monospace;">RECALL</p>
              <p style="font-size:1.1rem;font-weight:500;color:#e0e0e0;margin:0;
                        font-family:'JetBrains Mono',monospace;">{model_metrics['recall'] * 100:.2f}%</p>
            </div>
            <div style="background:#0a0a0a;padding:12px 16px;border-left:1px solid #111;">
              <p style="font-size:0.58rem;color:#333;letter-spacing:0.1em;margin:0 0 6px;
                        font-family:'JetBrains Mono',monospace;">F1 SCORE</p>
              <p style="font-size:1.1rem;font-weight:500;color:#e0e0e0;margin:0;
                        font-family:'JetBrains Mono',monospace;">{model_metrics['f1'] * 100:.2f}%</p>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        alert_banner(f"Unable to compute model metrics: {e}", "Warning")

    divider()

    summary_col, preview_col = st.columns([1, 2])
    with summary_col:
        section_label("Dataset Summary")
        st.metric("Rows", f"{metrics['rows']:,}")
        st.metric("Features", metrics["features"])
        st.metric("Missing Values", metrics["missing_values"])

    with preview_col:
        section_label("Data Preview")
        rows = st.selectbox("Select rows to display", [50, 100, 500, "All"])
        if rows == "All":
            alert_banner("Large dataset may slow UI. Showing a 1,000-row sample.", "Warning")
            display_df = df.sample(min(len(df), 1000), random_state=42)
        else:
            display_df = df.head(int(rows))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    divider()
    section_label("Latency Distribution")
    try:
        hist_df = prepare_visual_df(df_sample)
        fig, ax = plt.subplots(figsize=(6, 2.6), facecolor="#0f172a")
        ax.hist(hist_df["log_latency"].values, bins=40, color="#888888", alpha=0.92)
        style_dark_axes(ax, xlabel="Log Latency", ylabel="Records")
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)
    except Exception as e:
        alert_banner(f"Plot error: {e}", "Critical")


def network_analytics_tab(
    df: pd.DataFrame,
    df_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model,
    shap_sample: pd.DataFrame,
) -> None:
    """Render latency, throughput, and correlation analytics."""
    page_title("Analytics", "monitor")

    plot_df = prepare_visual_df(df_sample)
    plot_df = plot_df.assign(congestion_label=plot_df["congestion"].map({0: "Normal", 1: "Congested"}))

    col1, col2 = st.columns(2)
    with col1:
        section_label("Latency vs Throughput Density")
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
            st.pyplot(style_fig(fig), use_container_width=True)
            plt.close(fig)
        except Exception as e:
            alert_banner(f"Plot error: {e}", "Critical")

    with col2:
        section_label("Packet Loss vs Throughput Density")
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
            st.pyplot(style_fig(fig), use_container_width=True)
            plt.close(fig)
        except Exception as e:
            alert_banner(f"Plot error: {e}", "Critical")

    divider()

    col3, col4 = st.columns(2)
    with col3:
        section_label("Correlation Heatmap")
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
            st.pyplot(style_fig(fig), use_container_width=True)
            plt.close(fig)
        except Exception as e:
            alert_banner(f"Plot error: {e}", "Critical")

    with col4:
        section_label("Latency by Congestion State")
        try:
            box_df = clean_plot_df(plot_df, ["log_latency", "congestion_label"])
            fig, ax = plt.subplots(figsize=(6, 3.4), facecolor="#0f172a")
            sns.boxplot(
                data=box_df,
                x="congestion_label",
                y="log_latency",
                palette={"Normal": "#16a34a", "Congested": "#f97316"},
                ax=ax,
            )
            style_dark_axes(ax, xlabel="Congestion", ylabel="Log Latency")
            st.pyplot(style_fig(fig), use_container_width=True)
            plt.close(fig)
        except Exception as e:
            alert_banner(f"Plot error: {e}", "Critical")

    divider()
    section_label("Throughput Shape by Congestion")
    try:
        violin_df = clean_plot_df(plot_df, ["log_throughput", "congestion_label"])
        fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0f172a")
        sns.violinplot(
            data=violin_df,
            x="congestion_label",
            y="log_throughput",
            palette={"Normal": "#16a34a", "Congested": "#f97316"},
            cut=0,
            ax=ax,
        )
        style_dark_axes(ax, xlabel="Congestion", ylabel="Log Throughput")
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)
    except Exception as e:
        alert_banner(f"Plot error: {e}", "Critical")

    divider()
    section_label("Relationship Interpretation")
    try:
        corr_df = clean_plot_df(plot_df, FEATURE_COLUMNS)
        corr = corr_df[FEATURE_COLUMNS].corr(numeric_only=True)
        packet_throughput = corr.loc["packet_loss", "throughput"]
        latency_throughput = corr.loc["latency", "throughput"]

        if packet_throughput < -0.30:
            alert_banner("Higher packet loss strongly reduces throughput.", "Warning")
        elif packet_throughput > 0.30:
            st.info("Packet loss rises with throughput pressure, suggesting overloaded traffic windows.")
        else:
            st.info("Packet loss has a weak direct relationship with throughput in the sampled data.")

        if latency_throughput < -0.30:
            alert_banner("Higher latency is associated with lower throughput.", "Warning")
        elif latency_throughput > 0.30:
            st.info("Latency and throughput rise together, indicating heavier flows may be stressing the network.")
        else:
            st.info("Latency and throughput show a weak linear relationship, so congestion is likely multi-factor.")
    except Exception as e:
        alert_banner(f"Plot error: {e}", "Critical")

    divider()
    root_cause_panel(model, feature_importance)

    divider()
    section_label("SHAP Explainability Summary")
    show_shap_summary(model, shap_sample)


def time_intelligence_tab(df: pd.DataFrame) -> None:
    """Render rolling congestion trend and spike warnings."""
    page_title("Time Intel", "monitor")

    chart_df = df.head(min(len(df), VIZ_SAMPLE_SIZE))
    time_df = clean_plot_df(chart_df[["time", "congestion", "latency"]], ["time", "congestion", "latency"]).copy()
    if time_df.empty:
        alert_banner("No data available", "Warning")
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
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)
    except Exception as e:
        alert_banner(f"Plot error: {e}", "Critical")

    section_label("Latency Rolling Average")
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
        alert_banner("High congestion detected in recent period", "Warning")
    else:
        alert_banner("Recent period is below the high-congestion spike threshold", "Normal")


def render_model_drivers(predictor, feat_names):
    imp  = predictor.feature_importances_
    data = sorted(zip(feat_names, imp),
                  key=lambda x: x[1], reverse=True)
    max_v = max(v for _, v in data) or 1

    parts = [
        '<div style="border:1px solid #141414;'
        'border-radius:4px;padding:12px 16px;'
        'background:#0a0a0a;">',
        '<p style="font-size:9px;color:#1e1e1e;'
        'letter-spacing:0.14em;margin:0 0 10px;'
        "font-family:'JetBrains Mono',monospace;"
        '">MODEL DRIVERS</p>',
    ]
    for i, (feat, val) in enumerate(data, 1):
        bw = str(int(val / max_v * 60))
        parts += [
            '<div style="display:flex;align-items:'
            'center;gap:12px;padding:5px 0;'
            'border-bottom:1px solid #0f0f0f;">',
            '<span style="font-size:9px;color:#252525;'
            'min-width:16px;font-family:'
            "'JetBrains Mono',monospace;\">"
            + str(i).zfill(2) + '</span>',
            '<span style="font-size:11px;color:#555;'
            'min-width:100px;font-family:'
            "'JetBrains Mono',monospace;\">"
            + feat + '</span>',
            '<div style="height:2px;width:' + bw
            + 'px;background:#1e3a5f;'
            'border-radius:1px;"></div>',
            '<span style="font-size:9px;color:#333;'
            "font-family:'JetBrains Mono',monospace;\">"
            + str(round(val, 3)) + '</span>',
            '</div>',
        ]
    parts.append('</div>')
    st.markdown(''.join(parts), unsafe_allow_html=True)


def root_cause_panel(model, feature_importance: pd.DataFrame) -> None:
    """Render top model-driven root causes."""
    section_label("Root Cause Analysis")

    if feature_importance.empty:
        st.info("The loaded model does not expose feature_importances_.")
        return

    render_model_drivers(model, list(TRAIN_ORDER))

    try:
        section_label("Feature Importance")
        importance_by_feature = dict(zip(feature_importance["feature"], feature_importance["importance"].astype(float)))
        features = list(TRAIN_ORDER)
        importances = np.asarray([importance_by_feature[feature] for feature in features], dtype=float)
        sorted_idx = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_idx]
        sorted_values = [importances[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor('#0e0e0e')
        ax.set_facecolor('#0e0e0e')

        ax.barh(sorted_features, sorted_values,
                color='#1e3a5f', edgecolor='none', height=0.32)

        max_val = max(sorted_values)
        for i, (f, v) in enumerate(zip(sorted_features,
                                        sorted_values)):
            if v > max_val * 0.6:
                ax.text(v - max_val*0.02, i,
                        str(round(v,3)),
                        va='center', ha='right',
                        fontsize=7, color='#111',
                        fontfamily='monospace')
            else:
                ax.text(v + max_val*0.02, i,
                        str(round(v,3)),
                        va='center', ha='left',
                        fontsize=7, color='#333',
                        fontfamily='monospace')

        ax.set_xlim(0, max_val * 1.25)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='#2a2a2a', labelsize=8, length=0)
        ax.set_xlabel('importance score', fontsize=7,
                      color='#222', fontfamily='monospace')
        ax.yaxis.set_tick_params(labelcolor='#444')
        ax.grid(axis='x', color='#161616', linewidth=0.4, alpha=0.6)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.8)
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)
        st.markdown(
            '<p style="font-size:0.62rem;color:#1e1e1e;'
            'font-family:\'JetBrains Mono\',monospace;'
            'margin-top:4px;">'
            'global RF importance across training data &middot; '
            'not specific to current prediction</p>',
            unsafe_allow_html=True)
    except Exception as e:
        alert_banner(f"Plot error: {e}", "Critical")


def tower_optimization_tab(df: pd.DataFrame) -> str:
    """Render tower congestion optimization view."""
    page_title("Towers", "control")

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
            "#dc2626" if l > 75 else
            "#d97706" if l > 60 else
            "#16a34a" for l in loads
        ]
        bars = ax.bar(tower_names, loads, color=colors)
        ax.axhline(75, color="#dc2626", lw=0.8, ls="--", alpha=0.4, label="Overload threshold (75%)")
        ax.legend()
        ax.set_xlabel("Tower")
        ax.set_ylabel("Load (%)")
        ax.set_title(title)
        ax.set_ylim(0, 100)
        return fig

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = tower_load_chart(before_loads, "Tower Loads Before Redistribution")
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)
    with chart_col2:
        fig = tower_load_chart(after_loads, "Tower Loads After Redistribution")
        st.pyplot(style_fig(fig), use_container_width=True)
        plt.close(fig)

    divider()
    section_label("Observed Tower Congestion")

    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            plot_df = clean_plot_df(tower_summary, ["tower", "congestion_percentage"])
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0f172a")
            bar_colors = [
                "#dc2626" if value > 75 else "#d97706" if value > 60 else "#16a34a"
                for value in plot_df["congestion_percentage"].values
            ]
            ax.bar(plot_df["tower"].astype(str), plot_df["congestion_percentage"].values, color=bar_colors)
            ax.set_xlabel("Tower")
            ax.set_ylabel("Congestion (%)")
            ax.axhline(75, color="#dc2626", lw=0.8, ls="--", alpha=0.4, label="Overload threshold (75%)")
            ax.legend()
            ax.set_ylim(0, max(100, plot_df["congestion_percentage"].max() * 1.15))
            style_dark_axes(ax, xlabel="Tower", ylabel="Congestion (%)")
            st.pyplot(style_fig(fig), use_container_width=True)
            plt.close(fig)
        except Exception as e:
            alert_banner(f"Plot error: {e}", "Critical")

    with col2:
        st.metric("Most congested tower", f"Tower {most_congested}")
        if worst_congestion > 75:
            alert_banner(f"Tower {most_congested} is critically congested", "Critical")
        elif worst_congestion > 60:
            alert_banner(f"Tower {most_congested} needs monitoring", "Warning")
        else:
            alert_banner(f"Tower {most_congested} is the highest risk tower, but below critical thresholds", "Normal")
        st.dataframe(
            tower_summary[["rank", "tower", "records", "congestion_percentage"]],
            use_container_width=True,
            hide_index=True,
        )

    divider()
    section_label("Tower Action Plan")
    for _, row in tower_summary.iterrows():
        tower = row["tower"]
        congestion = float(row["congestion_percentage"])

        if congestion > 75:
            status = "Critical congestion"
            action = "Upgrade tower capacity and add a new node"
            alert_banner(f"Tower {tower} -> {status} -> {action}", "Critical")
        elif congestion > 60:
            status = "Moderate congestion"
            action = "Monitor usage and prepare load balancing"
            alert_banner(f"Tower {tower} -> {status} -> {action}", "Warning")
        else:
            status = "Stable load"
            action = "Monitor usage"
            alert_banner(f"Tower {tower} -> {status} -> {action}", "Normal")

    return str(most_congested)


def render_overview(df: pd.DataFrame, df_sample: pd.DataFrame, model) -> None:
    overview_tab(df, df_sample, model)


def render_analytics(
    df: pd.DataFrame,
    df_sample: pd.DataFrame,
    feature_importance: pd.DataFrame,
    model,
    shap_sample: pd.DataFrame,
) -> None:
    network_analytics_tab(df, df_sample, feature_importance, model, shap_sample)


def render_time_intel(df: pd.DataFrame) -> None:
    time_intelligence_tab(df)


def render_towers(df: pd.DataFrame) -> None:
    tower_optimization_tab(df)


def render_predict(df: pd.DataFrame, predictor: Predictor, feature_importance: pd.DataFrame) -> None:
    prediction_control_tab(df, predictor, feature_importance)


def prediction_control_tab(df: pd.DataFrame, predictor: Predictor, feature_importance: pd.DataFrame) -> None:
    """Render smart prediction and control panel."""
    st.markdown("""
    <p style="font-size:0.6rem;color:#252525;letter-spacing:0.08em;
              margin-bottom:14px;font-family:'JetBrains Mono',monospace;">
      NETOPT / <span style="color:#555;">predict & control</span>
    </p>""", unsafe_allow_html=True)

    def predict_dashboard_inputs(latency_value: float, throughput_value: float, packet_loss_value: float) -> tuple[int, float, list[list[float]]]:
        # Build input in EXACT training order
        feature_order = TRAIN_ORDER
        model_throughput_pressure = 1.0 - float(throughput_value)
        input_values  = {
            'latency':     float(latency_value),
            'throughput':  model_throughput_pressure,
            'packet_loss': float(packet_loss_value),
        }
        input_row = [[input_values[f] for f in feature_order]]
        proba = predictor.model.predict_proba(input_row)[0]

        # Always use index of class '1' for congestion prob
        class_list    = list(predictor.model.classes_)
        congestion_idx = class_list.index(1)
        cong_prob      = float(proba[congestion_idx])

        pred = int(cong_prob > 0.5)
        return pred, cong_prob, input_row

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

            _ = (summary, comparison, report_markdown, latency_improvement, throughput_gain)
            render_optimization_log(optimization_log, congestion_reduction)
        except Exception:
            alert_banner("Optimization engine unavailable", "Warning")

    def render_upload_mode() -> None:
        """Render batch upload prediction and optimization mode."""
        file_uploader = st.file_uploader("Upload CSV", type=["csv"])
        if not st.button("Run Batch Analysis", type="primary"):
            st.info("Upload a CSV with latency, throughput, and packet_loss columns.")
            return

        try:
            if file_uploader is None:
                alert_banner("Please upload a CSV file before running batch analysis.", "Critical")
                return

            upload_df = pd.read_csv(file_uploader)
            missing_columns = [feature for feature in TRAIN_ORDER if feature not in upload_df.columns]
            if missing_columns:
                alert_banner("Missing required columns: latency, throughput, packet_loss", "Critical")
                return

            if len(upload_df) > 500:
                st.info("Processing first 500 rows for performance")
                upload_df = upload_df.head(500).copy()

            results = []
            with st.spinner("Running batch analysis..."):
                for row_number, (_, row) in enumerate(upload_df.iterrows(), start=1):
                    row_latency = float(row["latency"])
                    row_throughput = float(row["throughput"])
                    row_packet_loss = float(row["packet_loss"])
                    row_tower_load = float(row["tower_load"]) if "tower_load" in upload_df.columns and pd.notna(row["tower_load"]) else 70.0
                    feature_order = TRAIN_ORDER
                    row_input = [[
                        float(row[f.capitalize()] if f.capitalize()
                              in row else row[f])
                        for f in feature_order
                    ]]
                    proba     = predictor.model.predict_proba(row_input)[0]
                    class_list = list(predictor.model.classes_)
                    congestion_idx = class_list.index(1)
                    cong_prob = float(proba[congestion_idx])
                    pred = int(cong_prob > 0.5)
                    prob = cong_prob
                    alert_level = alert_from_probability(prob)
                    network_state = network_state_from_probability(prob)
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
                "Download Batch Report (CSV)",
                data=results_df.to_csv(index=False),
                file_name="batch_optimization_report.csv",
                mime="text/csv",
            )

            congested_rows = results_df[results_df["Congestion_Prob"] > 0.5]

            if len(congested_rows) > 0:
                divider()
                section_label(f"{len(congested_rows)} Congested Rows Detected")
                st.caption(
                    "Click below to run AI optimization on all congested rows "
                    "and see the predicted post-optimization network state."
                )

                if st.button("Optimize Congested Rows", type="primary", key="batch_optimize_btn"):
                    st.session_state["run_batch_optimize"] = True

            if (
                st.session_state.get("run_batch_optimize", False)
                and len(congested_rows) > 0
                and (not OPTIMIZER_AVAILABLE or run_optimization_simulation is None)
            ):
                alert_banner(
                    "Optimizer module not found. "
                    "Make sure optimizer/simulation_runner.py exists.",
                    "Warning",
                )

            if (
                st.session_state.get("run_batch_optimize", False)
                and len(congested_rows) > 0
                and OPTIMIZER_AVAILABLE
                and run_optimization_simulation is not None
            ):
                with st.spinner("Running AI optimization on congested rows..."):

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
                    section_label("Optimization Results")
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
                        "Download Optimized Results (CSV)",
                        data=csv_opt,
                        file_name="optimized_network_results.csv",
                        mime="text/csv",
                        key="dl_opt_results",
                    )

                    # ── Reset button ──────────────────────────────────
                    if st.button("Reset Optimization", key="reset_opt"):
                        st.session_state["run_batch_optimize"] = False
                        st.rerun()
        except Exception as error:
            alert_banner(f"Batch analysis failed: {error}", "Critical")

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
        run_batch_analysis = st.button("Run Batch Analysis", type="primary")

        if run_batch_analysis:
            try:
                if file_uploader is None:
                    alert_banner("Please upload a CSV file before running batch analysis.", "Critical")
                    return

                upload_df = pd.read_csv(file_uploader)
                missing_columns = [feature for feature in TRAIN_ORDER if feature not in upload_df.columns]
                if missing_columns:
                    alert_banner("Missing required columns: latency, throughput, packet_loss", "Critical")
                    return

                if len(upload_df) > 500:
                    st.info("Processing first 500 rows for performance")
                    upload_df = upload_df.head(500).copy()

                results = []
                with st.spinner("Running batch analysis..."):
                    for row_number, (_, row) in enumerate(upload_df.iterrows(), start=1):
                        row_latency = float(row["latency"])
                        row_throughput = float(row["throughput"])
                        row_packet_loss = float(row["packet_loss"])
                        row_tower_load = float(row["tower_load"]) if "tower_load" in upload_df.columns and pd.notna(row["tower_load"]) else 70.0
                        feature_order = TRAIN_ORDER
                        row_input = [[
                            float(row[f.capitalize()] if f.capitalize()
                                  in row else row[f])
                            for f in feature_order
                        ]]
                        proba     = predictor.model.predict_proba(row_input)[0]
                        class_list = list(predictor.model.classes_)
                        congestion_idx = class_list.index(1)
                        cong_prob = float(proba[congestion_idx])
                        pred = int(cong_prob > 0.5)
                        prob = cong_prob
                        alert_level = alert_from_probability(prob)
                        network_state = network_state_from_probability(prob)
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
                alert_banner(f"Batch analysis failed: {error}", "Critical")

        if st.session_state["batch_results_df"] is None:
            st.info("Upload a CSV with latency, throughput, and packet_loss columns.")
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
            "Download Batch Report (CSV)",
            data=results_df.to_csv(index=False),
            file_name="batch_optimization_report.csv",
            mime="text/csv",
        )

        if n_cong > 0:
            divider()
            section_label(f"{n_cong} Congested Rows Detected")
            st.caption(
                "Click below to run AI optimization on all congested rows "
                "and see the predicted post-optimization network state."
            )
            if st.button("Optimize Congested Rows", type="primary", key="batch_optimize_btn"):
                st.session_state["run_batch_optimize"] = True

        if st.session_state.get("run_batch_optimize", False):
            if not OPTIMIZER_AVAILABLE or run_optimization_simulation is None:
                alert_banner("Optimizer module not found. Make sure optimizer/simulation_runner.py exists.", "Warning")
            elif st.session_state["opt_results_df"] is None:
                with st.spinner("Running AI optimization..."):
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

            section_label("Optimization Results")
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
                "Download Optimized Results (CSV)",
                data=csv_opt,
                file_name="optimized_network_results.csv",
                mime="text/csv",
                key="dl_opt_results",
            )

            if st.button("Reset Optimization", key="reset_opt"):
                st.session_state["run_batch_optimize"] = False
                st.session_state["opt_results_df"] = None
                st.rerun()

    mode = st.radio(
        "",
        ["Manual Input", "Upload Dataset"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.caption("Latency: 0–5000 model scale · Packet loss: 0–1 ratio · Throughput health: 0–1, where 1 is healthy")
    st.markdown('<div style="height:1px;background:#111;margin:14px 0 18px;"></div>', unsafe_allow_html=True)

    if mode == "Upload Dataset":
        render_upload_mode()
        return

    col_inputs, col_shap = st.columns([11, 9], gap="large")
    run_analysis = False
    pred = None
    prob = None
    input_row = None
    alert_level = None
    network_state = None

    with col_inputs:
        st.markdown(
            '<p style="font-size:9px;color:#252525;letter-spacing:0.14em;'
            'font-family:JetBrains Mono,monospace;margin-bottom:12px;">INPUTS</p>',
            unsafe_allow_html=True,
        )
        row1_left, row1_right = st.columns(2, gap="large")
        with row1_left:
            st.markdown(
                '<p style="font-size:9px;color:#252525;letter-spacing:0.12em;'
                'font-family:\'JetBrains Mono\',monospace;margin-bottom:4px;">LATENCY</p>',
                unsafe_allow_html=True,
            )
            latency = st.slider("", 0, 5000, 100, key="lat", label_visibility="collapsed")
            st.caption("Model scale · 0 – 5000")

        with row1_right:
            st.markdown(
                '<p style="font-size:9px;color:#252525;letter-spacing:0.12em;'
                'font-family:\'JetBrains Mono\',monospace;margin-bottom:4px;">PACKET LOSS</p>',
                unsafe_allow_html=True,
            )
            packet_loss = st.slider("", 0.0, 1.0, 0.01, step=0.01, key="pl", label_visibility="collapsed")
            st.caption("Ratio · 0.0 – 1.0")

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)

        row2_left, row2_right = st.columns(2, gap="large")
        with row2_left:
            st.markdown(
                '<p style="font-size:9px;color:#252525;letter-spacing:0.12em;'
                'font-family:\'JetBrains Mono\',monospace;margin-bottom:4px;">THROUGHPUT</p>',
                unsafe_allow_html=True,
            )
            throughput = st.slider("", 0.0, 1.0, 1.0, step=0.01, key="tp", label_visibility="collapsed")
            st.caption("Health ratio · 0.0 – 1.0")

        with row2_right:
            st.markdown(
                '<p style="font-size:9px;color:#252525;letter-spacing:0.12em;'
                'font-family:\'JetBrains Mono\',monospace;margin-bottom:4px;">TOWER LOAD</p>',
                unsafe_allow_html=True,
            )
            tower_load = st.slider("", 0.0, 100.0, 60.0, step=0.5, key="tl", label_visibility="collapsed")
            st.caption("Percent · 0 – 100")

        run_analysis = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
        if run_analysis:
            pred, prob, input_row = predict_dashboard_inputs(latency, throughput, packet_loss)
            network_state = network_state_from_probability(prob)
            alert_level = alert_from_probability(prob)

    with col_shap:
        st.markdown(
            '<p style="font-size:9px;color:#252525;letter-spacing:0.14em;'
            'font-family:JetBrains Mono,monospace;margin-bottom:12px;">'
            'FEATURE ATTRIBUTION · SHAP</p>',
            unsafe_allow_html=True,
        )
        if run_analysis and input_row is not None:
            show_shap_bar_chart(predictor.model, input_row)
        else:
            st.markdown("""
            <div style="height:180px;border:1px dashed #1a1a1a;border-radius:4px;
                        display:flex;align-items:center;justify-content:center;
                        margin-top:28px;">
              <span style="font-size:10px;color:#222;
                           font-family:'JetBrains Mono',monospace;">
                attribution map · run analysis first
              </span>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#111;margin:18px 0;"></div>', unsafe_allow_html=True)

    if run_analysis and prob is not None and alert_level is not None and network_state is not None:
        prediction_metric_grid(network_state, prob, alert_level)
        alert_banner(f"{alert_level} network state | Congestion probability: {prob:.1%}", alert_level)

        col_rca, col_sug = st.columns(2)
        with col_rca:
            section_label("Root Cause Analysis")
            for cause in input_root_causes(df, latency, throughput, packet_loss, int(pred) == 1):
                st.write(f"• {cause}")

        with col_sug:
            section_label("Suggested Actions")
            for suggestion in dynamic_suggestions(df, latency, throughput, packet_loss, prob, int(pred) == 1):
                st.markdown(f"- {suggestion}")

        if prob > 0.5:
            render_inline_optimization(latency, throughput, packet_loss, tower_load, prob)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="AI Network Performance Optimization",
        layout="wide",
    )

    if 'page' not in st.session_state:
        st.session_state['page'] = 'overview'

    st.markdown("""<style>
  @import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3.11.0/dist/tabler-icons.min.css');
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText, button,
  label, input, [data-testid] {
    font-family: 'JetBrains Mono', monospace !important;
  }
  p, label, .stMarkdown { color: #888 !important; }
  .stApp, [data-testid="stHeader"] { background: #0e0e0e !important; color: #e0e0e0 !important; }
  [data-testid="stAppViewContainer"],
  .main,
  .main .block-container {
    background-color: #0e0e0e !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E") !important;
    background-repeat: repeat !important;
    background-size: 128px !important;
  }
  .main .block-container { padding: 1.5rem 2.5rem 1.5rem 2rem; max-width: 1100px; }
  section[data-testid="stSidebar"] {
    min-width: 195px !important;
    max-width: 195px !important;
    background-color: #080808 !important;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E") !important;
    background-repeat: repeat !important;
    background-size: 128px !important;
    border-right: 1px solid #111 !important;
  }
  section[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
  }
  section[data-testid="stSidebar"] .block-container {
    padding: 12px 0 !important;
  }
  h1,h2,h3,h4 { font-family: 'JetBrains Mono', monospace !important; font-weight: 500 !important; letter-spacing: -0.02em !important; }
  [data-testid="metric-container"] {
    background: #0d0d0d !important; border: 1px solid #1e1e1e !important;
    border-radius: 4px !important; padding: 1rem !important;
  }
  [data-testid="metric-container"] label {
    font-size: 0.6rem !important; letter-spacing: 0.12em !important;
    text-transform: uppercase !important; color: #404040 !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.6rem !important; font-weight: 500 !important;
    color: #e0e0e0 !important;
  }
  .stButton > button {
    background: transparent !important; border: 1px solid #222 !important;
    border-radius: 3px !important; color: #888 !important;
    font-size: 0.68rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; padding: 6px 16px !important;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .stButton > button:hover { border-color: #444 !important; color: #ccc !important; background: #111 !important; }
  .stButton > button[kind="primary"] { border-color: #60a5fa !important; color: #60a5fa !important; }
  section[data-testid="stSidebar"] .stButton > button {
    height: 32px !important;
    width: 100% !important;
    cursor: pointer !important;
    background: transparent !important;
    border: none !important;
  }
  section[data-testid="stSidebar"] .stButton {
    margin-top: -32px !important;
    opacity: 0 !important;
    height: 32px !important;
    overflow: hidden !important;
  }
  section[data-testid="stSidebar"] .stButton > button:hover {
    background: transparent !important;
  }
  section[data-testid="stSidebar"] .stButton > button * {
    color: inherit !important;
    text-decoration: none !important;
  }
  section[data-testid="stSidebar"] a,
  section[data-testid="stSidebar"] a:visited,
  section[data-testid="stSidebar"] a:hover {
    color: #444 !important;
    text-decoration: none !important;
  }
  .stDataFrame { border: 1px solid #1e1e1e !important; border-radius: 4px !important; }
  .stDataFrame thead th { background: #080808 !important; font-size: 0.65rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #404040 !important; border-bottom: 1px solid #1a1a1a !important; }
  .stDataFrame tbody td { font-size: 0.75rem !important; color: #aaa !important; border-bottom: 1px solid #111 !important; }
  .stDataFrame tbody tr:hover td { background: #0f0f0f !important; }
  .streamlit-expanderHeader { font-size: 0.68rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; color: #555 !important; background: #0a0a0a !important; border: 1px solid #1e1e1e !important; border-radius: 3px !important; }
  hr { border-color: #111 !important; }
  .stRadio label { font-size: 0.75rem !important; color: #777 !important; }
  label[data-testid="stWidgetLabel"] { font-size: 0.65rem !important; color: #404040 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
  .stSlider > div > div > div {
    background: #1e1e1e !important;
  }
  .stSlider > div > div > div > div {
    background: #555 !important;
    width: 10px !important;
    height: 10px !important;
    border-radius: 50% !important;
  }
  [data-testid="stThumbValue"] {
    color: #ef4444 !important;
    font-size: 11px !important;
  }
  [data-testid="stFileUploader"] { border: 1px dashed #1a1a1a !important; border-radius: 4px !important; background: #080808 !important; }
  .stCaption, [data-testid="stCaptionContainer"] {
    color: #2a2a2a !important;
    font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
  }
  .stSpinner > div { border-top-color: #444 !important; }
  .stAlert { background: #0a0a0a !important; border-color: #1a1a1a !important; border-radius: 4px !important; }
</style>""", unsafe_allow_html=True)

    if not os.path.exists(MODEL_PATH):
        alert_banner("Model not found. Please run training first.", "Critical")
        st.stop()

    try:
        predictor = load_predictor()
    except Exception as error:
        alert_banner(f"Unable to load trained model from {MODEL_PATH}: {error}", "Critical")
        st.stop()

    try:
        df = load_dashboard_data()
    except Exception as error:
        alert_banner(f"Unable to load real data from data/raw: {error}", "Critical")
        st.stop()

    if df.empty:
        alert_banner("No data available", "Warning")
        st.stop()

    df_sample = sample_for_visualization(df)
    shap_sample = sample_for_visualization(df, SHAP_SAMPLE_SIZE)[FEATURE_COLUMNS]
    try:
        feature_importance = get_model_feature_importance(predictor)
    except PermissionError:
        feature_importance = pd.DataFrame(columns=["feature", "importance"])
        alert_banner("Feature importance disabled (Windows permission issue)", "Warning")

    render_sidebar_navigation()
    topbar()

    if st.session_state['page'] == 'overview':
        render_overview(df, df_sample, predictor.model)
    elif st.session_state['page'] == 'analytics':
        render_analytics(df, df_sample, feature_importance, predictor.model, shap_sample)
    elif st.session_state['page'] == 'time_intel':
        render_time_intel(df)
    elif st.session_state['page'] == 'towers':
        render_towers(df)
    elif st.session_state['page'] == 'predict':
        render_predict(df, predictor, feature_importance)



if __name__ == "__main__":
    main()
