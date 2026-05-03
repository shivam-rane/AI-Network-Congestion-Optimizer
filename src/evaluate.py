"""Model evaluation helpers and CI validation."""

import json
import sys

from src.config import METADATA_PATH


def validate_metrics(metrics, min_accuracy=0.80):
    """Raise an error when model metrics do not meet production gates."""
    if metrics["accuracy"] < min_accuracy:
        raise ValueError(f"Model accuracy {metrics['accuracy']:.4f} is below required {min_accuracy:.2f}")
    return True


def main():
    """Validate saved metadata for CI."""
    if not METADATA_PATH.exists():
        raise FileNotFoundError("models/metadata.json not found. Run training first.")

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    validate_metrics(metadata["metrics"], min_accuracy=0.80)
    print("Model validation passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error)
        sys.exit(1)
