# AI-based Network Congestion Detection System

This is a beginner-friendly Python project for detecting network congestion using machine learning and a Streamlit dashboard.

## Project Files

- `data_generation.py` creates a synthetic dataset with 10,000 network samples.
- `train_model.py` trains a `RandomForestClassifier` and saves the model.
- `app.py` runs the Streamlit dashboard.
- `requirements.txt` lists the required Python libraries.

## Setup

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

Generate the dataset:

```bash
python data_generation.py
```

Train the machine learning model:

```bash
python train_model.py
```

Start the dashboard:

```bash
streamlit run app.py
```

## Dashboard Features

- User input sliders for latency, throughput, packet loss, and jitter
- Predict button for live congestion detection
- Green output for normal network status
- Red output for congested network status
- Histogram of latency
- Scatter plot of latency vs packet loss
- Correlation heatmap
