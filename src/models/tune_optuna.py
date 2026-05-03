"""Lightweight Optuna tuning for Random Forest hyperparameters."""

import json

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_cicids_data
from src.features.feature_engineering import create_features
from src.models.train import balance_training_data, label_congestion
from src.utils.config import BEST_PARAMS_PATH, MODEL_FEATURES, PREDICTION_THRESHOLD


def objective(trial, feature_df):
    """Optuna objective optimized for F1-score."""
    train_df, test_df = train_test_split(feature_df, test_size=0.20, random_state=42, shuffle=True)
    latency_threshold = train_df["latency"].quantile(0.75)
    throughput_threshold = train_df["throughput"].quantile(0.75)

    y_train = label_congestion(train_df, latency_threshold, throughput_threshold).to_numpy()
    y_test = label_congestion(test_df, latency_threshold, throughput_threshold).to_numpy()
    train_labeled = balance_training_data(train_df, y_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_labeled[MODEL_FEATURES].to_numpy())
    y_train_balanced = train_labeled["congestion"].to_numpy()
    x_test = scaler.transform(test_df[MODEL_FEATURES].to_numpy())

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 250, step=50),
        "max_depth": trial.suggest_int("max_depth", 6, 14),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 1,
    }

    model = RandomForestClassifier(**params)
    model.fit(x_train, y_train_balanced)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities > PREDICTION_THRESHOLD).astype(int)
    return f1_score(y_test, predictions, zero_division=0)


def run_tuning(n_trials=20):
    """Run Optuna tuning and save best parameters."""
    raw_df, _, _ = load_cicids_data()
    feature_df = create_features(raw_df)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, feature_df), n_trials=n_trials)

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)

    print("Best F1-score:", study.best_value)
    print("Best params:", study.best_params)
    print("Saved best params to:", BEST_PARAMS_PATH)


if __name__ == "__main__":
    run_tuning(n_trials=20)
