"""Basic pipeline tests for schema and feature logic."""

import pandas as pd

from src.features import build_features, build_prediction_row


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
    row = build_prediction_row(0.5, 0.7, 2.0)

    assert row.shape[0] == 1
    assert "load_ratio" in row.columns
    assert "packet_intensity" in row.columns
