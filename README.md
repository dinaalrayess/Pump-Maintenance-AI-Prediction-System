#  AI-Based Predictive Maintenance System for Industrial Pumps

##  Overview

This project is an advanced AI-driven system designed to analyze, predict, and monitor industrial pump performance using machine learning techniques.

It combines predictive modeling, anomaly detection, and trend-based forecasting to simulate real-world maintenance decision systems used in the oil & gas and petrochemical industries.

---

##  Key Features

###  Predictive Maintenance

* Built multiple machine learning models (Random Forest, Gradient Boosting, Logistic Regression)
* Compared performance using ROC-AUC and cross-validation
* Selected best-performing model for failure prediction

### 🚨 Anomaly Detection

* Implemented Isolation Forest to detect abnormal pump behavior
* Identified ~5% anomalous operational patterns
* Provided per-pump anomaly analysis

###  Feature Engineering

* Developed engineered features including:

  * Sensor ratios (Temperature/Vibration, Pressure/Flow)
  * Per-pump z-score normalization
  * Wear index based on operational hours and vibration
* Enhanced model performance and interpretability

###  Trend-Based Forecasting

* Applied exponential smoothing to estimate future failure trends
* Used operational hour segmentation to simulate temporal behavior

### 📊 Data Visualization

* Generated:

  * ROC curves for model comparison
  * Feature importance rankings
  * Correlation heatmaps
  * Failure trend forecasts
  * Pump-level performance dashboards

---

##  Technologies Used

* Python
* NumPy, Pandas
* scikit-learn
* statsmodels
* Matplotlib & Seaborn

---

## 📊 Dataset

Industrial pump dataset including:

* Temperature
* Vibration
* Pressure
* Flow Rate
* RPM
* Operational Hours
* Maintenance Flag

---

## ⚙️ How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
python predictive_maintenance_ml.py
streamlit run dashboard.py
```

---

##  Applications

* Predictive maintenance in oil & gas
* Petrochemical plant monitoring
* Industrial equipment optimization
* Smart asset management systems

---
