"""Feature engineering for real CICIDS-based congestion modeling."""

import numpy as np
import pandas as pd

from src.preprocess import clean_raw_data, robust_minmax


def build_features(raw_df):
    """
    Build telecom performance features from real CICIDS flow metrics.

    packet_loss is derived from high-latency and low-throughput anomalies, not
    random generation.
    """
    df = clean_raw_data(raw_df)

    latency_norm = robust_minmax(df["Flow Duration"])
    throughput_norm = robust_minmax(df["Flow Bytes/s"])
    packet_rate_norm = robust_minmax(df["Flow Packets/s"])

    rolling_latency_mean = latency_norm.rolling(window=50, min_periods=1).mean()
    rolling_throughput_mean = throughput_norm.rolling(window=50, min_periods=1).mean()

    latency_anomaly = latency_norm
    low_throughput_anomaly = 1 - throughput_norm
    packet_loss = (10 * (0.65 * latency_anomaly + 0.35 * low_throughput_anomaly)).clip(0, 10)

    feature_df = pd.DataFrame()
    feature_df["latency_norm"] = latency_norm
    feature_df["throughput_norm"] = throughput_norm
    feature_df["packet_loss"] = packet_loss
    feature_df["rolling_latency_mean"] = rolling_latency_mean
    feature_df["rolling_throughput_mean"] = rolling_throughput_mean
    feature_df["load_ratio"] = throughput_norm / (latency_norm + 0.05)
    feature_df["packet_intensity"] = packet_loss * throughput_norm
    feature_df["time_index"] = np.arange(len(feature_df))
    feature_df["hour_of_day"] = (feature_df["time_index"] // 3600) % 24

    pressure_score = (
        0.35 * feature_df["latency_norm"]
        + 0.25 * feature_df["rolling_latency_mean"]
        + 0.20 * (feature_df["packet_loss"] / 10)
        + 0.20 * (1 - feature_df["rolling_throughput_mean"])
    )
    feature_df["network_pressure"] = pressure_score
    threshold = pressure_score.quantile(0.75)
    feature_df["congestion_flag"] = (pressure_score > threshold).astype(int)

    low_load = feature_df["throughput_norm"].quantile(0.33)
    high_load = feature_df["throughput_norm"].quantile(0.66)
    feature_df["tower"] = np.where(
        feature_df["throughput_norm"] > high_load,
        "B",
        np.where(feature_df["throughput_norm"] < low_load, "C", "A"),
    )

    return feature_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def build_prediction_row(latency_norm, throughput_norm, packet_loss, rolling_latency_mean=None, rolling_throughput_mean=None, hour_of_day=0):
    """Create a single model-ready prediction row."""
    rolling_latency_mean = latency_norm if rolling_latency_mean is None else rolling_latency_mean
    rolling_throughput_mean = throughput_norm if rolling_throughput_mean is None else rolling_throughput_mean
    return pd.DataFrame(
        [
            {
                "latency_norm": latency_norm,
                "throughput_norm": throughput_norm,
                "packet_loss": packet_loss,
                "rolling_latency_mean": rolling_latency_mean,
                "rolling_throughput_mean": rolling_throughput_mean,
                "load_ratio": throughput_norm / (latency_norm + 0.05),
                "packet_intensity": packet_loss * throughput_norm,
                "hour_of_day": hour_of_day,
            }
        ]
    )
