"""Training pipeline for the production-grade congestion model."""

import json
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from src.config import (
    BEST_PARAMS_PATH,
    DEFAULT_MODEL_PARAMS,
    LATEST_MODEL_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_FEATURES,
    MODEL_PATH,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    SCALER_PATH,
    TARGET_COLUMN,
)
from src.data_loader import load_cicids_dataset
from src.evaluate import validate_metrics
from src.features import build_features
from src.logging_utils import get_logger


def split_data(feature_df):
    """Create leakage-safe train/test split."""
    x = feature_df[MODEL_FEATURES]
    y = feature_df[TARGET_COLUMN]
    return train_test_split(x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)


def balance_training_frame(x_train, y_train):
    """Oversample minority class inside training data only."""
    train_df = x_train.copy()
    train_df[TARGET_COLUMN] = y_train.to_numpy()
    majority = train_df[train_df[TARGET_COLUMN] == 0]
    minority = train_df[train_df[TARGET_COLUMN] == 1]

    if len(minority) > 0 and len(minority) / len(train_df) < 0.35:
        target_size = int(len(majority) * 0.75)
        minority = resample(minority, replace=True, n_samples=target_size, random_state=RANDOM_STATE)
        train_df = pd.concat([majority, minority], ignore_index=True)

    return train_df[MODEL_FEATURES], train_df[TARGET_COLUMN]


def tune_hyperparameters(x_train, y_train, n_trials=20):
    """Tune RandomForest hyperparameters with Optuna for F1-score."""
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 260, step=20),
            "max_depth": trial.suggest_int("max_depth", 6, 14),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 15),
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="f1")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as file:
        json.dump(study.best_params, file, indent=2)
    return study.best_params


def train_model(feature_df, tune=False):
    """Train model and return artifacts plus validation metrics."""
    x_train, x_test, y_train, y_test = split_data(feature_df)
    x_train_balanced, y_train_balanced = balance_training_frame(x_train, y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_balanced)
    x_test_scaled = scaler.transform(x_test)

    params = DEFAULT_MODEL_PARAMS.copy()
    if tune:
        tuned = tune_hyperparameters(x_train_scaled, y_train_balanced, n_trials=20)
        params.update(tuned)
        params["class_weight"] = "balanced"
        params["random_state"] = RANDOM_STATE
        params["n_jobs"] = 1

    model = RandomForestClassifier(**params)
    model.fit(x_train_scaled, y_train_balanced)

    probabilities = model.predict_proba(x_test_scaled)[:, 1]
    predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics["cross_val_f1"] = float(cross_val_score(model, x_train_scaled, y_train_balanced, cv=cv, scoring="f1").mean())

    feature_importance = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    dataset_probabilities = model.predict_proba(scaler.transform(feature_df[MODEL_FEATURES]))[:, 1]
    return model, scaler, metrics, feature_importance, dataset_probabilities


def save_artifacts(model, scaler, metrics):
    """Save model, scaler, and metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    versioned_model = MODEL_DIR / f"model_{timestamp}.pkl"

    joblib.dump(model, MODEL_PATH)
    joblib.dump(model, LATEST_MODEL_PATH)
    joblib.dump(model, versioned_model)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "created_at": timestamp,
        "prediction_threshold": PREDICTION_THRESHOLD,
        "features": MODEL_FEATURES,
        "metrics": metrics,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


def run_training(tune=False, save=True):
    """Run the full data-to-model training pipeline."""
    logger = get_logger()
    logger.info("Training started")
    raw_df, loaded_rows, original_columns = load_cicids_dataset()
    feature_df = build_features(raw_df)
    model, scaler, metrics, feature_importance, dataset_probabilities = train_model(feature_df, tune=tune)
    validate_metrics(metrics, min_accuracy=0.80)

    if save:
        save_artifacts(model, scaler, metrics)

    logger.info(
        "Training finished rows=%s cols=%s accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        loaded_rows,
        original_columns,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    return {
        "feature_df": feature_df,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "dataset_probabilities": dataset_probabilities,
        "loaded_rows": loaded_rows,
        "original_columns": original_columns,
    }


def main():
    """CLI entrypoint."""
    result = run_training(tune=False, save=True)
    metrics = result["metrics"]
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
