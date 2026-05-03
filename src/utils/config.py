"""Central project configuration."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
LOG_DIR = ROOT_DIR / "logs"

CICIDS_DATA_DIR = Path(r"C:\Users\shiva\Downloads\archive (1)")
CICIDS_FILES = [
    CICIDS_DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv",
    CICIDS_DATA_DIR / "Wednesday-workingHours.pcap_ISCX.csv",
]

RAW_FEATURES = ["Flow Duration", "Flow Bytes/s", "Flow Packets/s"]
MODEL_FEATURES = ["latency", "throughput", "packet_loss", "load_ratio", "packet_intensity"]
TARGET_COLUMN = "congestion"
MAX_ROWS = 50_000
PREDICTION_THRESHOLD = 0.65

DEFAULT_MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
}

LATEST_MODEL_PATH = MODEL_DIR / "latest_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"
