"""
AI-Based Predictive Maintenance System for Industrial Pumps
============================================================
Uses: Large_Industrial_Pump_Maintenance_Dataset.csv

Install dependencies:
    pip install pandas numpy scikit-learn matplotlib seaborn statsmodels

Run:
    python predictive_maintenance_ml.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ─────────────────────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  PREDICTIVE MAINTENANCE — INDUSTRIAL PUMPS")
print("=" * 60)

df = pd.read_csv("Large_Industrial_Pump_Maintenance_Dataset.csv")

print(f"\n[DATA] Loaded {len(df):,} records, {df.shape[1]} features")
print(f"[DATA] Pumps: {sorted(df['Pump_ID'].unique())}")
print(f"[DATA] Maintenance rate: {df['Maintenance_Flag'].mean()*100:.1f}%")
print("\n--- Dataset Overview ---")
print(df.describe().round(2))

# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n[FEATURES] Engineering new features...")

# Interaction features
df["Temp_Vib_ratio"]     = df["Temperature"] / (df["Vibration"] + 0.001)
df["Pressure_Flow_ratio"]= df["Pressure"]    / (df["Flow_Rate"] + 0.001)
df["RPM_normalized"]     = df["RPM"]         / df["RPM"].max()
df["Hours_normalized"]   = df["Operational_Hours"] / df["Operational_Hours"].max()

# Anomaly signal: deviation from per-pump mean
for col in ["Temperature", "Vibration", "Pressure", "Flow_Rate", "RPM"]:
    pump_mean = df.groupby("Pump_ID")[col].transform("mean")
    pump_std  = df.groupby("Pump_ID")[col].transform("std")
    df[f"{col}_zscore"] = (df[col] - pump_mean) / (pump_std + 1e-9)

# Wear indicator (hours × vibration)
df["Wear_Index"] = df["Operational_Hours"] * df["Vibration"] / 10000

print(f"[FEATURES] Total features after engineering: {df.shape[1]}")

# ─────────────────────────────────────────────────────────────
# 3. ANOMALY DETECTION — ISOLATION FOREST
# ─────────────────────────────────────────────────────────────
print("\n[ANOMALY] Running Isolation Forest...")

sensor_cols = ["Temperature", "Vibration", "Pressure", "Flow_Rate",
               "RPM", "Operational_Hours"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[sensor_cols])

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # expect ~5% anomalies
    random_state=42,
    n_jobs=-1
)
df["anomaly_score"] = iso.fit_predict(X_scaled)
df["anomaly_raw"]   = iso.decision_function(X_scaled)
df["is_anomaly"]    = (df["anomaly_score"] == -1).astype(int)

n_anomalies = df["is_anomaly"].sum()
print(f"[ANOMALY] Detected {n_anomalies:,} anomalies ({n_anomalies/len(df)*100:.1f}%)")

# Per-pump anomaly breakdown
print("\n--- Anomaly Rate by Pump ---")
pump_anomaly = df.groupby("Pump_ID").agg(
    total=("is_anomaly", "count"),
    anomalies=("is_anomaly", "sum"),
    anomaly_rate=("is_anomaly", "mean"),
    avg_anomaly_score=("anomaly_raw", "mean"),
    maint_rate=("Maintenance_Flag", "mean")
).round(4)
pump_anomaly["anomaly_pct"] = (pump_anomaly["anomaly_rate"] * 100).round(2)
print(pump_anomaly[["total", "anomalies", "anomaly_pct", "maint_rate"]])

# ─────────────────────────────────────────────────────────────
# 4. PREDICTIVE MODEL — FAILURE CLASSIFICATION
# ─────────────────────────────────────────────────────────────
print("\n[MODEL] Training failure prediction models...")

feature_cols = sensor_cols + [
    "Temp_Vib_ratio", "Pressure_Flow_ratio", "RPM_normalized",
    "Hours_normalized", "Wear_Index",
    "Temperature_zscore", "Vibration_zscore", "Pressure_zscore",
    "Flow_Rate_zscore", "RPM_zscore",
    "is_anomaly"
]

X = df[feature_cols]
y = df["Maintenance_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

models = {
    "Random Forest":        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=500, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc",
                         n_jobs=-1) if name != "Logistic Regression" else \
         cross_val_score(model, X_train_sc, y_train, cv=5, scoring="roc_auc")

    auc = roc_auc_score(y_test, y_prob)
    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "auc": auc, "cv_mean": cv.mean(), "cv_std": cv.std()
    }
    print(f"  {name:25s}  AUC={auc:.4f}  CV={cv.mean():.4f}±{cv.std():.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["auc"])
best = results[best_name]
print(f"\n[MODEL] Best model: {best_name} (AUC={best['auc']:.4f})")

print("\n--- Classification Report (Best Model) ---")
print(classification_report(y_test, best["y_pred"], target_names=["Normal", "Maintenance"]))

# Feature importances (Random Forest)
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\n--- Top 10 Feature Importances (Random Forest) ---")
print(importances.head(10).round(4))

# ─────────────────────────────────────────────────────────────
# 5. TIME-SERIES FAILURE FORECAST
# ─────────────────────────────────────────────────────────────
print("\n[FORECAST] Building time-series failure rate forecast...")

# Simulate monthly maintenance rate by binning operational hours into 12 buckets
df["hours_bin"] = pd.cut(df["Operational_Hours"], bins=12, labels=False)
monthly_rate = df.groupby("hours_bin")["Maintenance_Flag"].mean().reset_index()
monthly_rate.columns = ["period", "failure_rate"]

# Holt-Winters exponential smoothing
ts = monthly_rate["failure_rate"].values
model_hw = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
fit_hw   = model_hw.fit(optimized=True)
forecast = fit_hw.forecast(6)

print("[FORECAST] Next 6-period failure rate predictions:")
for i, v in enumerate(forecast, 1):
    risk = "🔴 HIGH" if v > 0.55 else ("🟡 MEDIUM" if v > 0.45 else "🟢 LOW")
    print(f"  Period +{i}: {v:.3f} ({v*100:.1f}%) — {risk}")

# Per-pump failure probability using best model
print("\n--- Current Failure Probability per Pump ---")
for pump_id in sorted(df["Pump_ID"].unique()):
    pump_data = df[df["Pump_ID"] == pump_id][feature_cols].tail(50)
    if best_name == "Logistic Regression":
        pump_sc = scaler.transform(pump_data)
        prob = results[best_name]["model"].predict_proba(pump_sc)[:, 1].mean()
    else:
        prob = results[best_name]["model"].predict_proba(pump_data)[:, 1].mean()
    risk = "🔴 HIGH" if prob > 0.55 else ("🟡 MEDIUM" if prob > 0.45 else "🟢 LOW")
    print(f"  Pump {pump_id}: {prob*100:.1f}% failure probability — {risk}")

# ─────────────────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
print("\n[VIZ] Generating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Predictive Maintenance — Industrial Pumps", fontsize=16, fontweight="bold")
fig.patch.set_facecolor("#0d1520")
for ax in axes.flatten():
    ax.set_facecolor("#111d2e")
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("#00d4ff"); ax.spines[:].set_color("#1a3050")

ACCENT = "#00d4ff"; GREEN = "#00ff9d"; WARN = "#ffd93d"; DANGER = "#ff3d3d"

# Plot 1: Maintenance rate by pump
ax = axes[0, 0]
pump_rates = df.groupby("Pump_ID")["Maintenance_Flag"].mean() * 100
colors_bar = [DANGER if v > 50 else WARN if v > 48 else GREEN for v in pump_rates]
ax.bar([f"P-{p}" for p in pump_rates.index], pump_rates.values, color=colors_bar, edgecolor="#0d1520", linewidth=0.5)
ax.axhline(y=49.8, color=ACCENT, linestyle="--", linewidth=1, label="Fleet avg")
ax.set_title("MAINTENANCE RATE BY PUMP"); ax.set_ylabel("Rate (%)"); ax.set_ylim(45, 55)
ax.legend(facecolor="#0d1520", labelcolor="white")

# Plot 2: Anomaly score distribution
ax = axes[0, 1]
for pump_id in sorted(df["Pump_ID"].unique()):
    subset = df[df["Pump_ID"] == pump_id]["anomaly_raw"]
    ax.hist(subset, bins=40, alpha=0.5, label=f"P-{pump_id}", density=True)
ax.axvline(x=0, color=DANGER, linestyle="--", linewidth=1.5, label="Threshold")
ax.set_title("ISOLATION FOREST ANOMALY SCORES"); ax.set_xlabel("Anomaly Score")
ax.legend(facecolor="#0d1520", labelcolor="white", fontsize=8)

# Plot 3: Feature importances
ax = axes[0, 2]
top_feat = importances.head(8)
ax.barh(top_feat.index[::-1], top_feat.values[::-1], color=ACCENT, edgecolor="#0d1520")
ax.set_title("TOP 8 FEATURE IMPORTANCES (RF)"); ax.set_xlabel("Importance")

# Plot 4: ROC curves
ax = axes[1, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    c = ACCENT if name == best_name else "#4a6a8a"
    lw = 2 if name == best_name else 1
    ax.plot(fpr, tpr, color=c, linewidth=lw, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1], "w--", linewidth=0.5)
ax.set_title("ROC CURVES — ALL MODELS"); ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(facecolor="#0d1520", labelcolor="white", fontsize=8)

# Plot 5: Sensor correlation heatmap
ax = axes[1, 1]
corr_matrix = df[sensor_cols + ["Maintenance_Flag"]].corr()
mask = np.zeros_like(corr_matrix, dtype=bool); mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, ax=ax,
            annot=True, fmt=".2f", annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8}, linecolor="#0d1520", linewidths=0.3)
ax.set_title("SENSOR CORRELATION MATRIX")
ax.tick_params(colors="white", labelsize=7)

# Plot 6: Failure forecast
ax = axes[1, 2]
periods_hist = list(range(len(ts)))
periods_fore = list(range(len(ts), len(ts) + 6))
ax.plot(periods_hist, ts * 100, color=ACCENT, linewidth=2, label="Historical")
ax.plot(periods_hist, fit_hw.fittedvalues * 100, color=GREEN, linewidth=1, linestyle="--", label="Fitted")
ax.plot(periods_fore, forecast * 100, color=DANGER, linewidth=2, marker="o", markersize=4, label="Forecast")
ax.axhline(y=55, color=DANGER, linestyle=":", linewidth=1, label="Risk threshold 55%")
ax.set_title("FAILURE RATE FORECAST (HOLT-WINTERS)"); ax.set_xlabel("Period"); ax.set_ylabel("Failure Rate (%)")
ax.legend(facecolor="#0d1520", labelcolor="white", fontsize=8)

plt.tight_layout()
plt.savefig("maintenance_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1520")
plt.show()
print("[VIZ] Saved → maintenance_analysis.png")

# ─────────────────────────────────────────────────────────────
# 7. MAINTENANCE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  MAINTENANCE RECOMMENDATIONS")
print("=" * 60)

for pump_id in sorted(df["Pump_ID"].unique()):
    pump_data = df[df["Pump_ID"] == pump_id]
    maint_rate = pump_data["Maintenance_Flag"].mean()
    avg_vib    = pump_data["Vibration"].mean()
    avg_temp   = pump_data["Temperature"].mean()
    n_anomalies_p = pump_data["is_anomaly"].sum()

    risk = "HIGH" if maint_rate > 0.50 else ("MEDIUM" if maint_rate > 0.48 else "LOW")
    print(f"\n  Pump {pump_id} [{risk} RISK]")
    print(f"    Maintenance rate : {maint_rate*100:.1f}%")
    print(f"    Avg vibration    : {avg_vib:.2f} mm/s")
    print(f"    Avg temperature  : {avg_temp:.1f} °C")
    print(f"    Anomalies found  : {n_anomalies_p}")
    if maint_rate > 0.50:
        print(f"    → ACTION: Schedule inspection within 2 weeks")
    elif avg_vib > 2.54:
        print(f"    → ACTION: Monitor vibration closely, check bearings")
    else:
        print(f"    → ACTION: Continue routine monitoring")

print("\n[DONE] Analysis complete.")
