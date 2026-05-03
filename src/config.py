"""Project-wide configuration for the CICIDS congestion optimizer."""

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = Path(os.getenv("CICIDS_DATA_DIR", DATA_DIR / "raw"))
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

RAW_COLUMNS = [
    "Flow Duration",
    "Flow Bytes/s",
    "Flow Packets/s",
]

MODEL_FEATURES = [
    "latency_norm",
    "throughput_norm",
    "packet_loss",
    "rolling_latency_mean",
    "rolling_throughput_mean",
    "load_ratio",
    "packet_intensity",
    "hour_of_day",
]

TARGET_COLUMN = "congestion_flag"
MAX_ROWS = 50_000
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.65

MODEL_PATH = MODEL_DIR / "model.pkl"
LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}
