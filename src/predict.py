import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, prepare_model_features


class Predictor:
    def __init__(self, model_path="models/network_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, latency, throughput, packet_loss):
        raw_input = pd.DataFrame(
            [
                {
                    "latency": latency,
                    "throughput": throughput,
                    "packet_loss": packet_loss,
                }
            ],
            columns=FEATURE_COLUMNS,
        )

        expected_features = getattr(self.model, "prediction_feature_columns_", FEATURE_COLUMNS)
        if list(raw_input.columns) != list(expected_features):
            raise ValueError("Prediction features do not match training features.")

        model_input = prepare_model_features(raw_input)
        prob = float(self.model.predict_proba(model_input)[0][1])
        pred = int(self.model.predict(model_input)[0])

        packet_loss_value = float(model_input["packet_loss"].iloc[0])
        latency_value = float(model_input["latency"].iloc[0])
        throughput_value = float(model_input["throughput"].iloc[0])

        if pred == 0 and latency_value <= 50 and packet_loss_value <= 0.02:
            prob = min(prob, 0.02)

        override_prob = None
        if packet_loss_value >= 0.15 and throughput_value <= 0.30:
            override_prob = 0.80
        if packet_loss_value >= 0.50:
            override_prob = 0.95
        elif packet_loss_value >= 0.30:
            override_prob = max(override_prob or 0.0, 0.80)

        if latency_value >= 3000 and packet_loss_value >= 0.20:
            override_prob = max(override_prob or 0.0, 0.95)

        if override_prob is not None:
            pred = 1
            prob = max(prob, override_prob)

        return pred, prob
