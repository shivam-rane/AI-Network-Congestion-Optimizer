"""Prediction helpers for dashboard and CLI use."""

import joblib

from src.features.feature_engineering import create_prediction_features
from src.utils.config import LATEST_MODEL_PATH, MODEL_FEATURES, PREDICTION_THRESHOLD, SCALER_PATH


def load_artifacts(model_path=LATEST_MODEL_PATH, scaler_path=SCALER_PATH):
    """Load latest saved model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_with_model(model, scaler, input_df, threshold=PREDICTION_THRESHOLD):
    """Predict congestion using an already-loaded model and scaler."""
    x = scaler.transform(input_df[MODEL_FEATURES].to_numpy())
    probability = float(model.predict_proba(x)[0][1])
    prediction = int(probability > threshold)
    return prediction, probability


def predict_congestion(latency, throughput, packet_loss, threshold=PREDICTION_THRESHOLD):
    """Load latest artifacts and predict one network condition."""
    model, scaler = load_artifacts()
    input_df = create_prediction_features(latency, throughput, packet_loss)
    return predict_with_model(model, scaler, input_df, threshold=threshold)
