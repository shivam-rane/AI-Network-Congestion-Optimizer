import joblib
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
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
    X_scaled = scaler.transform(X)

    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET (unseen data)")
    print("="*60)
    
    # Evaluate on test set only
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1 Score:  {test_f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # Cross-validation on training set
    print("\n" + "="*60)
    print("CROSS-VALIDATION (5-fold on training data)")
    print("="*60)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\n" + "="*60)
    print("Saving model and scaler...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/network_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("✅ Model saved at models/network_model.pkl")
    print("✅ Scaler saved at models/scaler.pkl")
    print("="*60)

    print("\n✓ Training complete.")


if __name__ == "__main__":
    train()
