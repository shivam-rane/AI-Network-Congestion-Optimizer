# 🚀 AI Telecom Network Congestion Optimizer

An AI-powered network intelligence system that detects, predicts, explains, and supports optimization decisions for telecom network congestion using real-world traffic data.

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
* Suggest actionable optimizations and load-balancing actions

---

## 🎯 Key Features

* 📊 Real-time network analytics dashboard
* 🔮 Dynamic ML-based congestion prediction
* ⚠️ Probability-based alerts (Normal / Warning / Critical)
* 🧠 Root cause analysis (data-driven)
* 💡 Dynamic recommendations and optimization suggestions
* 🗼 Tower load balancing simulation and optimization
* 📉 Explainable AI using SHAP
* ⚙️ Threshold-tuned predictions with rule-based safety upgrades

---

## 🧠 System Architecture

```mermaid
graph LR
    A["📥 Raw CICIDS Data"] --> B["🧹 Data Cleaning"]
    B --> C["⚙️ Feature Engineering"]
    C --> D["🤖 Model Training<br/>Random Forest"]
    D --> E["💾 Model Registry<br/>.pkl Storage"]
    E --> F["🎯 Prediction Engine"]
    F --> G["📊 Streamlit Dashboard"]
    G --> H["📈 Monitoring &<br/>Insights"]
    
    style A fill:#1e40af,stroke:#1e3a8a,color:#fff
    style B fill:#2563eb,stroke:#1e40af,color:#fff
    style C fill:#3b82f6,stroke:#2563eb,color:#fff
    style D fill:#7c3aed,stroke:#6d28d9,color:#fff
    style E fill:#ec4899,stroke:#be185d,color:#fff
    style F fill:#f59e0b,stroke:#d97706,color:#fff
    style G fill:#10b981,stroke:#059669,color:#fff
    style H fill:#06b6d4,stroke:#0891b2,color:#fff
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

Random Forest is used because telecom congestion patterns are often nonlinear and depend on interactions between latency, throughput, and packet loss. Tree ensembles can capture these feature interactions without requiring a strictly linear decision boundary, making them suitable for congestion prediction where multiple moderate signals can combine into a high-risk network state.

* Input features:

  * latency
  * throughput
  * packet_loss

* Target:

  * congestion (from CICIDS labels with packet-loss safety handling)

* Prediction approach:

  * probability-based congestion scoring
  * threshold-tuned classification
  * balanced evaluation with precision, recall, and F1 score
  * safety upgrades for high-risk packet-loss, latency, and throughput conditions

---

## 📈 Model Performance

| Metric    | Value              |
| --------- | ------------------ |
| Accuracy  | 96.92%             |
| Precision | 90.47%             |
| Recall    | 95.51%             |
| F1 Score  | 92.92%             |

Evaluation uses a clean 80/20 stratified train/test split with `random_state=99` for test evaluation, separate from the model training seed. Metrics are reported only on unseen test data to avoid data leakage and keep performance realistic.

👉 Model selection balances overall accuracy with congestion recall, so the system avoids missing risky congestion cases while still maintaining strong precision.

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

Dynamic recommendations based on network state and prediction probability:

* High latency → optimize routing
* High packet loss → improve signal quality
* Low throughput → increase bandwidth
* High congestion → load balancing

---

## 🗼 Tower Optimization

* Simulates multiple network towers
* Identifies overloaded nodes
* Suggests traffic redistribution and load-balancing strategies

---

## 📊 Dashboard Features

* Overview (KPIs + model metrics)
* Network Analytics (correlation, distributions)
* Time Intelligence (trend + spikes)
* Tower Optimization
* Prediction & Control panel with dynamic alerts and recommendations

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
