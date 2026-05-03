"""Feature engineering for telecom congestion modeling."""

import numpy as np
import pandas as pd

from src.data.preprocess import clean_raw_data


def create_features(raw_df):
    """Create model-ready telecom features from raw CICIDS flow columns."""
    df = clean_raw_data(raw_df)

    duration_clipped = df["Flow Duration"].clip(0, df["Flow Duration"].quantile(0.99))
    bytes_clipped = df["Flow Bytes/s"].clip(0, df["Flow Bytes/s"].quantile(0.99))
    packets_clipped = df["Flow Packets/s"].clip(0, df["Flow Packets/s"].quantile(0.99))

    duration_norm = duration_clipped / max(duration_clipped.max(), 1)
    bytes_norm = bytes_clipped / max(bytes_clipped.max(), 1)
    packets_norm = packets_clipped / max(packets_clipped.max(), 1)

    traffic_pressure = 0.45 * duration_norm + 0.35 * bytes_norm + 0.20 * packets_norm

    features = pd.DataFrame()
    features["latency"] = (0.75 * traffic_pressure + 0.25 * duration_norm) * 300
    features["throughput"] = (0.75 * traffic_pressure + 0.25 * bytes_norm) * bytes_clipped.max()
    features["packet_loss"] = (100 / (df["Flow Packets/s"] + 10)).clip(0, 10)
    features["load_ratio"] = features["throughput"] / (features["latency"] + 1)
    features["packet_intensity"] = features["packet_loss"] * features["throughput"]
    features["time"] = range(len(features))

    low_traffic = features["throughput"].quantile(0.33)
    high_traffic = features["throughput"].quantile(0.66)
    features["tower"] = np.where(
        features["throughput"] > high_traffic,
        "B",
        np.where(features["throughput"] < low_traffic, "C", "A"),
    )

    return features.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def create_prediction_features(latency, throughput, packet_loss):
    """Create one prediction row with the same engineered features as training."""
    load_ratio = throughput / (latency + 1)
    packet_intensity = packet_loss * throughput
    return pd.DataFrame(
        [
            {
                "latency": latency,
                "throughput": throughput,
                "packet_loss": packet_loss,
                "load_ratio": load_ratio,
                "packet_intensity": packet_intensity,
            }
        ]
    )
