import joblib
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import load_data
from src.features import create_features


def train():
    print("Loading data...")
    df = load_data()

    print("Creating features...")
    df = create_features(df)

    features = ["latency", "throughput", "packet_loss"]
    X = df[features]
    y = df["congestion"]

    scaler = StandardScaler()
    scaler.fit(X)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=150, max_depth=10)
    model.fit(X_train, y_train)

    print("Evaluating...")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/network_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model saved at models/network_model.pkl")

    print("Training complete.")


if __name__ == "__main__":
    train()
