# AI Network Congestion Optimizer

Production-style ML and MLOps project for predicting and optimizing telecom network congestion using the real CICIDS 2017 network-flow dataset.

## Problem Statement

Telecom networks need early warning systems that can detect congestion risk before service quality degrades. This project converts real CICIDS network-flow metrics into telecom performance signals, predicts near-future congestion, explains root causes, and recommends operational actions.

## Overview

The system uses real CICIDS CSV files only. It does not generate synthetic traffic. The pipeline loads network-flow data, cleans missing and invalid values, engineers telecom features, trains a Random Forest model, saves artifacts, and serves predictions through a Streamlit dashboard.

## Architecture

```text
Real CICIDS CSVs
      |
      v
src/data_loader.py
      |
      v
src/preprocess.py
      |
      v
src/features.py
      |
      v
src/train.py ----> models/model.pkl, models/scaler.pkl, models/metadata.json
      |
      v
src/evaluate.py
      |
      v
dashboard/app.py
```

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── dashboard/
│   └── app.py
├── docs/images/
├── tests/
│   └── test_pipeline.py
├── .github/workflows/
│   └── train.yml
├── requirements.txt
├── README.md
└── app.py
```

## Dataset

Use the CICIDS 2017 CSV files. Place them in:

```text
data/raw/
```

The loader reads only the required real flow columns:

- `Flow Duration`
- `Flow Bytes/s`
- `Flow Packets/s`

You can also point to an external local folder:

```bash
set CICIDS_DATA_DIR=C:\path\to\cicids_csv_folder
```

## Feature Engineering

The system creates:

- normalized latency
- normalized throughput
- packet loss derived from high-latency and low-throughput anomalies
- rolling latency mean
- rolling throughput mean
- load ratio
- packet intensity
- hour-of-day feature
- near-future congestion flag

The target predicts congestion ahead of the current flow window, reducing leakage and making the task more realistic.

## Model

Default model:

```text
RandomForestClassifier(
    n_estimators=200,
    max_depth=14,
    min_samples_leaf=5,
    class_weight="balanced"
)
```

The pipeline uses:

- train/test split
- StandardScaler
- class imbalance handling by oversampling training data
- cross-validation
- Optuna tuning support
- model artifact saving

## Current Metrics

Latest local run on 50k CICIDS rows:

| Metric | Value |
|---|---:|
| Accuracy | 85.77% |
| Precision | 67.85% |
| Recall | 81.83% |
| F1-score | 74.19% |

These are validation metrics, not training scores.

## Dashboard

Run:

```bash
streamlit run app.py
```

Dashboard features:

- smart probability-based prediction
- Normal / Warning / Critical alert levels
- root cause analysis
- dynamic suggestions
- early warning spike detection
- tower load optimization
- clean network visualizations

Screenshots should be saved in:

```text
docs/images/
```

## MLOps Pipeline

Train:

```bash
python -m src.train
```

Evaluate saved model gate:

```bash
python -m src.evaluate
```

Run Optuna tuning:

```bash
python -m src.tune_optuna
```

Predict from code:

```python
from src.predict import predict

prediction, probability = predict(
    latency_norm=0.7,
    throughput_norm=0.8,
    packet_loss=4.0,
)
```

## CI/CD

GitHub Actions workflow:

```text
.github/workflows/train.yml
```

On push or pull request, it installs dependencies, trains the model, and fails if validation accuracy is below 80%.

## How To Run Locally

```bash
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

## Testing

```bash
pytest
```

## Future Improvements

- Replace Random Forest with XGBoost or LightGBM
- Add MLflow experiment tracking
- Add drift monitoring
- Store model artifacts in cloud storage
- Add scheduled retraining
- Add real tower/site metadata
