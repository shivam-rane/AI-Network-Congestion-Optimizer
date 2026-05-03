"""Production inference helpers."""

import json

import joblib

from src.config import METADATA_PATH, MODEL_FEATURES, MODEL_PATH, PREDICTION_THRESHOLD, SCALER_PATH
from src.features import build_prediction_row


def load_model_artifacts(model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    """Load trained model, scaler, and metadata if available."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    return model, scaler, metadata


def predict_from_features(model, scaler, feature_row, threshold=PREDICTION_THRESHOLD):
    """Predict one row and return prediction plus probability."""
    scaled = scaler.transform(feature_row[MODEL_FEATURES])
    probability = float(model.predict_proba(scaled)[0][1])
    prediction = int(probability >= threshold)
    return prediction, probability


def predict(latency_norm, throughput_norm, packet_loss, rolling_latency_mean=None, rolling_throughput_mean=None, hour_of_day=0):
    """Load artifacts and predict congestion for one network state."""
    model, scaler, metadata = load_model_artifacts()
    threshold = metadata.get("prediction_threshold", PREDICTION_THRESHOLD)
    feature_row = build_prediction_row(
        latency_norm,
        throughput_norm,
        packet_loss,
        rolling_latency_mean=rolling_latency_mean,
        rolling_throughput_mean=rolling_throughput_mean,
        hour_of_day=hour_of_day,
    )
    return predict_from_features(model, scaler, feature_row, threshold=threshold)
