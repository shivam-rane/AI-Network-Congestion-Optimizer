# 🚀 AI Telecom Network Congestion Optimizer

An AI-powered network intelligence system that detects, predicts, and explains congestion in telecom networks using real-world traffic data.

---

## 📌 Problem Statement

Telecom networks frequently suffer from:

* High latency
* Packet loss
* Traffic congestion
* Poor resource utilization

Traditional systems are **reactive** and lack predictive intelligence.

👉 This project builds a **proactive AI system** to:

* Detect congestion early
* Predict network conditions
* Explain root causes
* Suggest actionable optimizations

---

## 🎯 Key Features

* 📊 Real-time network analytics dashboard
* 🔮 ML-based congestion prediction
* ⚠️ Early warning system (Normal / Warning / Critical)
* 🧠 Root cause analysis (data-driven)
* 💡 Dynamic optimization suggestions
* 🗼 Tower load simulation and optimization
* 📉 Explainable AI using SHAP

---

## 🧠 System Architecture

```text
Raw CICIDS Data
      ↓
Data Loading & Cleaning
      ↓
Feature Engineering
      ↓
Model Training (Random Forest)
      ↓
Prediction Engine
      ↓
Streamlit Dashboard
      ↓
Insights + Recommendations
```

---

## 📂 Dataset

* **CICIDS 2017 Dataset**
* Real-world network traffic data

### Key Raw Features Used:

* Flow Duration
* Total Forward Packets
* Total Backward Packets
* Flow Bytes/s

---

## ⚙️ Feature Engineering

Since telecom KPIs are not directly available, features are derived:

* **Latency** → Flow Duration / Total Packets
* **Throughput** → Flow Bytes per second
* **Packet Loss (approx)** → imbalance between forward/backward packets

---

## 🤖 Model Details

* Model: **Random Forest Classifier**

* Input features:

  * latency
  * throughput
  * packet_loss

* Target:

  * congestion (derived from thresholds)

---

## 📈 Model Performance

| Metric    | Value              |
| --------- | ------------------ |
| Accuracy  | ~85–92%            |
| Precision | Balanced           |
| Recall    | High (prioritized) |
| F1 Score  | Balanced           |

👉 Model is tuned to **avoid missing congestion events (high recall)**

---

## 🔍 Explainability (SHAP)

The model uses SHAP to explain predictions:

* Identifies key drivers of congestion
* Provides transparent decision-making
* Enables actionable insights

Example:

* High packet loss → major contributor
* Increased latency → secondary driver

---

## ⚠️ Early Warning System

Based on prediction probability:

* 🟢 Normal (< 0.4)
* 🟡 Warning (0.4 – 0.7)
* 🔴 Critical (> 0.7)

---

## 💡 Intelligent Suggestion Engine

Dynamic recommendations based on network state:

* High latency → optimize routing
* High packet loss → improve signal quality
* Low throughput → increase bandwidth
* High congestion → load balancing

---

## 🗼 Tower Optimization

* Simulates multiple network towers
* Identifies overloaded nodes
* Suggests traffic redistribution strategies

---

## 📊 Dashboard Features

* Overview (KPIs + model metrics)
* Network Analytics (correlation, distributions)
* Time Intelligence (trend + spikes)
* Tower Optimization
* Prediction & Control panel

---

## ⚙️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* SHAP

---

## 🚀 How to Run

```bash
git clone https://github.com/shivam-rane/AI-Network-Congestion-Optimizer.git
cd AI-Network-Congestion-Optimizer

pip install -r requirements.txt

streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```text
.
├── dashboard/
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
├── data/
│   └── raw/   (ignored in Git)
├── models/
├── tests/
├── README.md
```

---

## ⚠️ Limitations

* Packet loss is approximated (not directly measured)
* Dataset is static (no real-time streaming yet)
* Model may require retraining for different network environments

---

## 🚀 Future Improvements

* Real-time streaming pipeline (Kafka / Spark)
* Advanced anomaly detection
* Deep learning models (LSTM for time series)
* Cloud deployment (AWS/GCP)

---

## 👨‍💻 Author

**Shivam Rane**

---

## ⭐ If you found this useful, consider starring the repo!
