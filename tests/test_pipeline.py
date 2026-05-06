"""Basic pipeline tests for schema and feature logic."""

import pandas as pd

from src.features import FEATURE_COLUMNS, build_features, build_prediction_row, create_features


def test_feature_pipeline_schema():
    """Feature pipeline should return expected model-ready columns."""
    raw_df = pd.DataFrame(
        {
            "Flow Duration": [100, 200, 300, 400, 500],
            "Flow Bytes/s": [1000, 900, 700, 500, 300],
            "Flow Packets/s": [10, 15, 20, 25, 30],
        }
    )

    features = build_features(raw_df)

    assert "latency_norm" in features.columns
    assert "throughput_norm" in features.columns
    assert "packet_loss" in features.columns
    assert "congestion_flag" in features.columns
    assert features["congestion_flag"].isin([0, 1]).all()


def test_prediction_row_schema():
    """Prediction rows should include engineered features."""
    row = build_prediction_row(0.5, 0.7, 0.2)

    assert row.shape[0] == 1
    assert "load_ratio" in row.columns
    assert "packet_intensity" in row.columns


def test_create_features_preserves_model_feature_signal():
    """Production features should stay varied and aligned with congestion risk."""
    raw_df = pd.DataFrame(
        {
            "Flow Duration": [100, 120, 150, 250, 5000, 8000, 12000, 16000],
            "Total Fwd Packets": [10, 12, 14, 18, 80, 100, 150, 200],
            "Total Backward Packets": [10, 11, 12, 15, 30, 20, 5, 0],
            "Flow Bytes/s": [100000, 90000, 75000, 60000, 2000, 1000, 300, 50],
        }
    )

    features = create_features(raw_df)

    assert list(features[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert features["latency"].between(0, 5000).all()
    assert features["throughput"].between(0, 1).all()
    assert features["packet_loss"].between(0, 1).all()
    assert features[FEATURE_COLUMNS].nunique().gt(1).all()
    assert features.loc[7, "packet_loss"] > features.loc[0, "packet_loss"]

    grouped = features.groupby("congestion")[FEATURE_COLUMNS].mean()
    assert grouped.loc[1, "latency"] > grouped.loc[0, "latency"]
    assert grouped.loc[1, "throughput"] < grouped.loc[0, "throughput"]
    assert grouped.loc[1, "packet_loss"] > grouped.loc[0, "packet_loss"]
