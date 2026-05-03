"""CLI entrypoint for lightweight Optuna hyperparameter tuning."""

import json

from src.config import BEST_PARAMS_PATH
from src.data_loader import load_cicids_dataset
from src.features import build_features
from src.train import split_data, balance_training_frame, tune_hyperparameters


def main():
    """Run 20 Optuna trials and save best parameters."""
    raw_df, _, _ = load_cicids_dataset()
    feature_df = build_features(raw_df)
    x_train, _, y_train, _ = split_data(feature_df)
    x_train_balanced, y_train_balanced = balance_training_frame(x_train, y_train)
    best_params = tune_hyperparameters(x_train_balanced, y_train_balanced, n_trials=20)

    BEST_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as file:
        json.dump(best_params, file, indent=2)

    print("Best parameters:", best_params)
    print("Saved to:", BEST_PARAMS_PATH)


if __name__ == "__main__":
    main()
