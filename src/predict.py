import joblib
import pandas as pd


class Predictor:
    def __init__(self, model_path="models/network_model.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, latency, throughput, packet_loss):
        data = pd.DataFrame([{
            "latency": latency,
            "throughput": throughput,
            "packet_loss": packet_loss
        }])

        pred = self.model.predict(data)[0]
        prob = self.model.predict_proba(data)[0][1]

        return pred, prob