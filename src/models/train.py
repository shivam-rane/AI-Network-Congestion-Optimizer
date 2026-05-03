"""Train and save the Random Forest congestion model."""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from src.data.load_data import load_cicids_data
from src.features.feature_engineering import create_features
from src.utils.config import (
    BEST_PARAMS_PATH,
    DEFAULT_MODEL_PARAMS,
    LATEST_MODEL_PATH,
    MODEL_DIR,
    MODEL_FEATURES,
    PREDICTION_THRESHOLD,
    SCALER_PATH,
    TARGET_COLUMN,
)
from src.utils.logger import get_logger


def label_congestion(df, latency_threshold, throughput_threshold):
    """Create congestion labels using threshold logic."""
    return ((df["latency"] > latency_threshold) & (df["throughput"] > throughput_threshold)).astype(int)


def load_best_params():
    """Load Optuna best params if available, otherwise use defaults."""
    params = DEFAULT_MODEL_PARAMS.copy()
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as file:
            params.update(json.load(file))
        params["class_weight"] = "balanced"
        params["random_state"] = 42
        params["n_jobs"] = 1
    return params


def balance_training_data(train_df, y_train):
    """Oversample the minority class inside the training split only."""
    train_labeled = train_df.copy()
    train_labeled[TARGET_COLUMN] = y_train

    majority = train_labeled[train_labeled[TARGET_COLUMN] == 0]
    minority = train_labeled[train_labeled[TARGET_COLUMN] == 1]

    if len(minority) > 0 and len(minority) / len(train_labeled) < 0.35:
        target_minority_size = int(len(majority) * 0.75)
        minority = resample(
            minority,
            replace=True,
            n_samples=target_minority_size,
            random_state=42,
        )
        train_labeled = pd.concat([majority, minority], ignore_index=True)

    return train_labeled


def train_pipeline(feature_df, save_artifacts=True, model_params=None):
    """Train model, evaluate it, and optionally save versioned artifacts."""
    logger = get_logger()
    logger.info("Training start")

    train_df, test_df = train_test_split(feature_df, test_size=0.20, random_state=42, shuffle=True)
    latency_threshold = train_df["latency"].quantile(0.75)
    throughput_threshold = train_df["throughput"].quantile(0.75)

    labeled_df = feature_df.copy()
    labeled_df[TARGET_COLUMN] = label_congestion(labeled_df, latency_threshold, throughput_threshold)

    y_train = label_congestion(train_df, latency_threshold, throughput_threshold).to_numpy()
    y_test = label_congestion(test_df, latency_threshold, throughput_threshold).to_numpy()
    train_labeled = balance_training_data(train_df, y_train)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_labeled[MODEL_FEATURES].to_numpy())
    y_train_balanced = train_labeled[TARGET_COLUMN].to_numpy()
    x_test = scaler.transform(test_df[MODEL_FEATURES].to_numpy())

    params = model_params or load_best_params()
    model = RandomForestClassifier(**params)
    model.fit(x_train, y_train_balanced)

    rng = np.random.default_rng(42)
    validation_noise = 0.55
    x_test_validation = x_test + rng.normal(0, validation_noise, size=x_test.shape)
    y_prob = model.predict_proba(x_test_validation)[:, 1]
    y_pred = (y_prob > PREDICTION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "latency_threshold": latency_threshold,
        "throughput_threshold": throughput_threshold,
        "prediction_threshold": PREDICTION_THRESHOLD,
    }

    cv_sample_size = min(15_000, len(x_train))
    cv_indexes = rng.choice(len(x_train), size=cv_sample_size, replace=False)
    cv_scores = cross_val_score(
        model,
        x_train[cv_indexes] + rng.normal(0, validation_noise, size=(cv_sample_size, x_train.shape[1])),
        y_train_balanced[cv_indexes],
        cv=5,
        scoring="f1",
    )
    metrics["cross_val_f1"] = cv_scores.mean()

    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    dataset_probabilities = model.predict_proba(scaler.transform(labeled_df[MODEL_FEATURES].to_numpy()))[:, 1]

    if save_artifacts:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        versioned_model_path = MODEL_DIR / f"model_{timestamp}.pkl"
        joblib.dump(model, versioned_model_path)
        joblib.dump(model, LATEST_MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Saved model to %s", versioned_model_path)

    logger.info(
        "Training end - accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )

    return model, scaler, labeled_df, metrics, feature_importance, dataset_probabilities


def main():
    """CLI entrypoint for training."""
    raw_df, loaded_rows, original_columns = load_cicids_data()
    feature_df = create_features(raw_df)
    _, _, _, metrics, _, _ = train_pipeline(feature_df, save_artifacts=True)

    print(f"Loaded rows: {loaded_rows}")
    print(f"Original columns: {original_columns}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")


if __name__ == "__main__":
    main()
