"""
Streamlit Predictive Maintenance Dashboard
===========================================
Install:
    pip install streamlit pandas numpy scikit-learn plotly statsmodels

Run:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Pump Predictive Maintenance",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #080c10; }
  .block-container { padding-top: 1.5rem; }
  h1, h2, h3 { color: #00d4ff !important; font-family: 'Barlow Condensed', sans-serif; letter-spacing: 2px; }
  .metric-card {
    background: #0d1520; border: 1px solid #1a3050; border-radius: 10px;
    padding: 16px 20px; text-align: center;
  }
  .metric-value { font-size: 32px; font-weight: 900; color: #00d4ff; }
  .metric-label { font-size: 11px; color: #4a6a8a; letter-spacing: 2px; margin-top: 4px; }
  .risk-high { color: #ff3d3d !important; font-weight: bold; }
  .risk-med  { color: #ffd93d !important; font-weight: bold; }
  .risk-low  { color: #00ff9d !important; font-weight: bold; }
  div[data-testid="stMetricValue"] { font-size: 28px !important; color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

DARK_BG   = "#080c10"
PANEL_BG  = "#0d1520"
ACCENT    = "#00d4ff"
GREEN     = "#00ff9d"
WARN      = "#ffd93d"
DANGER    = "#ff3d3d"

# ── LOAD & CACHE DATA ────────────────────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df

@st.cache_data
def engineer_features(df):
    df = df.copy()
    df["Temp_Vib_ratio"]      = df["Temperature"] / (df["Vibration"] + 0.001)
    df["Pressure_Flow_ratio"] = df["Pressure"]    / (df["Flow_Rate"] + 0.001)
    df["RPM_normalized"]      = df["RPM"]         / df["RPM"].max()
    df["Hours_normalized"]    = df["Operational_Hours"] / df["Operational_Hours"].max()
    df["Wear_Index"]          = df["Operational_Hours"] * df["Vibration"] / 10000
    for col in ["Temperature", "Vibration", "Pressure", "Flow_Rate", "RPM"]:
        pump_mean = df.groupby("Pump_ID")[col].transform("mean")
        pump_std  = df.groupby("Pump_ID")[col].transform("std")
        df[f"{col}_zscore"] = (df[col] - pump_mean) / (pump_std + 1e-9)
    return df

@st.cache_resource
def run_isolation_forest(df, sensor_cols, contamination):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(df[sensor_cols])
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42, n_jobs=-1)
    scores = iso.fit_predict(X_sc)
    raw    = iso.decision_function(X_sc)
    return scores, raw

@st.cache_resource
def train_rf(df, feature_cols):
    X = df[feature_cols]
    y = df["Maintenance_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return rf, auc, importances, X_test, y_test, y_pred, y_prob

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Industrial Pump AI Maintenance System")
    st.markdown("---")
    data_path = st.text_input("Dataset path", value="Large_Industrial_Pump_Maintenance_Dataset.csv")
    contamination = st.slider("Isolation Forest contamination", 0.01, 0.15, 0.05, 0.01,
                              help="Expected fraction of anomalies in the data")
    selected_pumps = st.multiselect("Filter pumps", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    st.markdown("---")
    st.markdown("**Dataset Stats**")
    st.caption("20,000 records · 5 pumps · 6 sensors")
    st.caption("Maintenance rate: ~49.8%")
    st.markdown("---")
    st.markdown("**Models Used**")
    st.caption("• Isolation Forest (anomaly)")
    st.caption("• Random Forest (classification)")
    st.caption("• Holt-Winters (forecasting)")

# ── LOAD DATA ────────────────────────────────────────────────
try:
    df_raw = load_data(data_path)
except FileNotFoundError:
    st.error(f"❌ File not found: `{data_path}`\n\nMake sure `Large_Industrial_Pump_Maintenance_Dataset.csv` is in the same folder.")
    st.stop()

df = engineer_features(df_raw)
df = df[df["Pump_ID"].isin(selected_pumps)].copy()

SENSOR_COLS = ["Temperature", "Vibration", "Pressure", "Flow_Rate", "RPM", "Operational_Hours"]
FEATURE_COLS = SENSOR_COLS + [
    "Temp_Vib_ratio", "Pressure_Flow_ratio", "RPM_normalized",
    "Hours_normalized", "Wear_Index",
    "Temperature_zscore", "Vibration_zscore", "Pressure_zscore",
    "Flow_Rate_zscore", "RPM_zscore",
]

# Run models
with st.spinner("Running Isolation Forest + Random Forest..."):
    scores, raw = run_isolation_forest(df, SENSOR_COLS, contamination)
    df["anomaly_score"] = scores
    df["anomaly_raw"]   = raw
    df["is_anomaly"]    = (scores == -1).astype(int)

    feat_cols_full = FEATURE_COLS + ["is_anomaly"]
    rf, auc, importances, X_test, y_test, y_pred, y_prob = train_rf(df, feat_cols_full)

# ── HEADER ───────────────────────────────────────────────────
st.markdown("#  INDUSTRIAL PUMPS — PREDICTIVE MAINTENANCE SYSTEM")
st.markdown("*AI-powered anomaly detection · failure prediction · time-series forecasting*")
st.markdown("---")

# ── KPI ROW ──────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(" Records",     f"{len(df):,}")
c2.metric(" Pumps",       len(df["Pump_ID"].unique()))
c3.metric("⚠️ Anomalies",   f"{df['is_anomaly'].sum():,}")
c4.metric(" Maint. Rate", f"{df['Maintenance_Flag'].mean()*100:.1f}%")
c5.metric(" RF AUC",      f"{auc:.4f}")
c6.metric(" Sensors",     "6")

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([" Overview", "⚠️ Anomaly Detection", " Failure Prediction", "📈 Forecast", " Pump Report"])

# ─────────────────────────────────────────────────────────────
# TAB 1: OVERVIEW
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("FLEET SENSOR OVERVIEW")

    col1, col2 = st.columns(2)

    with col1:
        # Maintenance rate by pump
        pump_summary = df.groupby("Pump_ID").agg(
            Maint_Rate=("Maintenance_Flag", "mean"),
            Avg_Temp=("Temperature", "mean"),
            Avg_Vibration=("Vibration", "mean"),
            Avg_RPM=("RPM", "mean"),
            Records=("Pump_ID", "count")
        ).reset_index()
        pump_summary["Maint_Rate_pct"] = pump_summary["Maint_Rate"] * 100

        fig = px.bar(pump_summary, x="Pump_ID", y="Maint_Rate_pct",
                     color="Maint_Rate_pct",
                     color_continuous_scale=["#00ff9d", "#ffd93d", "#ff3d3d"],
                     range_color=[48, 52],
                     title="Maintenance Rate by Pump (%)",
                     labels={"Pump_ID": "Pump", "Maint_Rate_pct": "Maintenance Rate (%)"},
                     text_auto=".1f")
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font_color="white", title_font_color=ACCENT, showlegend=False)
        fig.add_hline(y=49.8, line_dash="dash", line_color=ACCENT, annotation_text="Fleet avg")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Sensor distributions
        sensor = st.selectbox("Select sensor for distribution", SENSOR_COLS, key="dist_sensor")
        fig2 = px.histogram(df, x=sensor, color="Maintenance_Flag",
                            barmode="overlay",
                            color_discrete_map={0: GREEN, 1: DANGER},
                            title=f"{sensor} Distribution by Maintenance Flag",
                            labels={"Maintenance_Flag": "Needs Maintenance"},
                            opacity=0.75)
        fig2.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font_color="white", title_font_color=ACCENT)
        st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    st.subheader("SENSOR CORRELATION MATRIX")
    corr = df[SENSOR_COLS + ["Maintenance_Flag"]].corr().round(3)
    fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r",
                     title="Sensor Correlations (including Maintenance Flag)")
    fig3.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font_color="white", title_font_color=ACCENT)
    st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 2: ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("ISOLATION FOREST ANOMALY DETECTION")

    col1, col2 = st.columns(2)

    with col1:
        # Anomaly score distribution
        fig = px.histogram(df, x="anomaly_raw", color="is_anomaly",
                           color_discrete_map={0: GREEN, 1: DANGER},
                           nbins=60, barmode="overlay",
                           title="Anomaly Score Distribution",
                           labels={"anomaly_raw": "Anomaly Score", "is_anomaly": "Is Anomaly"})
        fig.add_vline(x=0, line_dash="dash", line_color=WARN, annotation_text="Threshold")
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font_color="white", title_font_color=ACCENT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Anomaly by pump
        anom_by_pump = df.groupby("Pump_ID").agg(
            total=("is_anomaly", "count"),
            anomalies=("is_anomaly", "sum"),
        ).reset_index()
        anom_by_pump["normal"] = anom_by_pump["total"] - anom_by_pump["anomalies"]
        anom_by_pump["anomaly_pct"] = (anom_by_pump["anomalies"] / anom_by_pump["total"] * 100).round(2)

        fig2 = go.Figure()
        fig2.add_bar(x=[f"P-{p}" for p in anom_by_pump["Pump_ID"]], y=anom_by_pump["normal"], name="Normal", marker_color=GREEN)
        fig2.add_bar(x=[f"P-{p}" for p in anom_by_pump["Pump_ID"]], y=anom_by_pump["anomalies"], name="Anomaly", marker_color=DANGER)
        fig2.update_layout(barmode="stack", title="Normal vs Anomaly Counts by Pump",
                           paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font_color="white", title_font_color=ACCENT)
        st.plotly_chart(fig2, use_container_width=True)

    # 2D scatter: anomalies in sensor space
    st.subheader("ANOMALY VISUALIZATION IN SENSOR SPACE")
    col_x = st.selectbox("X axis", SENSOR_COLS, index=0)
    col_y = st.selectbox("Y axis", SENSOR_COLS, index=1)
    sample = df.sample(min(3000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x=col_x, y=col_y,
                      color=sample["is_anomaly"].map({0: "Normal", 1: "Anomaly"}),
                      color_discrete_map={"Normal": GREEN, "Anomaly": DANGER},
                      opacity=0.5, title=f"Anomaly Map: {col_x} vs {col_y}")
    fig3.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font_color="white", title_font_color=ACCENT)
    st.plotly_chart(fig3, use_container_width=True)

    # Anomaly table
    st.subheader("TOP ANOMALY RECORDS")
    top_anomalies = df[df["is_anomaly"] == 1].nsmallest(20, "anomaly_raw")[
        ["Pump_ID"] + SENSOR_COLS + ["anomaly_raw", "Maintenance_Flag"]
    ].reset_index(drop=True)
    st.dataframe(top_anomalies.style.background_gradient(subset=["anomaly_raw"], cmap="RdYlGn"),
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB 3: FAILURE PREDICTION
# ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("RANDOM FOREST FAILURE PREDICTION")

    col1, col2 = st.columns(2)

    with col1:
        # Feature importance
        top_imp = importances.head(12)
        fig = px.bar(x=top_imp.values, y=top_imp.index,
                     orientation="h",
                     title=f"Top Feature Importances (RF AUC = {auc:.4f})",
                     color=top_imp.values,
                     color_continuous_scale=["#4a6a8a", ACCENT])
        fig.update_layout(yaxis=dict(autorange="reversed"),
                          paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font_color="white", title_font_color=ACCENT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Failure probability by pump
        pump_probs = []
        for pump_id in sorted(df["Pump_ID"].unique()):
            pump_data = df[df["Pump_ID"] == pump_id][feat_cols_full].tail(100)
            prob = rf.predict_proba(pump_data)[:, 1].mean()
            pump_probs.append({"Pump": f"P-{pump_id}", "Failure_Prob": prob * 100})
        pump_probs_df = pd.DataFrame(pump_probs)

        fig2 = px.bar(pump_probs_df, x="Pump", y="Failure_Prob",
                      color="Failure_Prob",
                      color_continuous_scale=["#00ff9d", "#ffd93d", "#ff3d3d"],
                      range_color=[45, 55],
                      title="Current Failure Probability by Pump (%)",
                      text_auto=".1f")
        fig2.add_hline(y=50, line_dash="dash", line_color=WARN, annotation_text="50% risk line")
        fig2.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                           font_color="white", title_font_color=ACCENT, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Classification report
    st.subheader("MODEL PERFORMANCE")
    report = classification_report(y_test, y_pred, target_names=["Normal", "Maintenance"], output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

    # Live prediction tool
    st.subheader(" LIVE FAILURE PREDICTOR")
    st.caption("Enter sensor readings to predict maintenance need:")
    lc1, lc2, lc3 = st.columns(3)
    inp_temp  = lc1.slider("Temperature (°C)",    50.0, 150.0, 100.0)
    inp_vib   = lc1.slider("Vibration (mm/s)",      0.1,   5.0,   2.5)
    inp_press = lc2.slider("Pressure (PSI)",       100.0, 300.0, 200.0)
    inp_flow  = lc2.slider("Flow Rate (L/s)",        0.5,  20.0,  10.0)
    inp_rpm   = lc3.slider("RPM",                 1000.0,3000.0,2000.0)
    inp_hours = lc3.slider("Operational Hours",    100.0,9999.0,5000.0)

    if st.button(" PREDICT FAILURE RISK", type="primary"):
        row = {
            "Temperature": inp_temp, "Vibration": inp_vib, "Pressure": inp_press,
            "Flow_Rate": inp_flow, "RPM": inp_rpm, "Operational_Hours": inp_hours
        }
        sample_df = pd.DataFrame([row])
        sample_df["Pump_ID"] = 1
        sample_df = engineer_features(sample_df)
        # anomaly score
        sc = StandardScaler()
        sc.fit(df[SENSOR_COLS])
        a_score = IsolationForest(n_estimators=100, contamination=contamination, random_state=42).fit(
            sc.transform(df[SENSOR_COLS])).decision_function(sc.transform(sample_df[SENSOR_COLS]))[0]
        sample_df["is_anomaly"] = 1 if a_score < 0 else 0

        prob = rf.predict_proba(sample_df[feat_cols_full])[0, 1]
        risk = "🔴 HIGH RISK" if prob > 0.55 else ("🟡 MEDIUM RISK" if prob > 0.45 else "🟢 LOW RISK")
        st.metric("Failure Probability", f"{prob*100:.1f}%", delta=risk)
        if prob > 0.55:
            st.error(f"⚠️ HIGH RISK — Schedule maintenance immediately! Probability: {prob*100:.1f}%")
        elif prob > 0.45:
            st.warning(f"⚡ MEDIUM RISK — Monitor closely. Probability: {prob*100:.1f}%")
        else:
            st.success(f"✅ LOW RISK — Normal operation. Probability: {prob*100:.1f}%")

# ─────────────────────────────────────────────────────────────
# TAB 4: TIME-SERIES FORECAST
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("FAILURE RATE FORECASTING — HOLT-WINTERS")

    df["hours_bin"] = pd.cut(df["Operational_Hours"], bins=12, labels=False)
    monthly = df.groupby("hours_bin")["Maintenance_Flag"].mean().reset_index()
    ts_vals  = monthly["Maintenance_Flag"].values

    n_periods = st.slider("Forecast periods ahead", 3, 12, 6)

    try:
        hw = ExponentialSmoothing(ts_vals, trend="add", seasonal=None, initialization_method="estimated")
        fit = hw.fit(optimized=True)
        forecast_vals = fit.forecast(n_periods)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(ts_vals))), y=ts_vals * 100,
                                 name="Historical", line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=list(range(len(ts_vals))), y=fit.fittedvalues * 100,
                                 name="Fitted", line=dict(color=GREEN, dash="dash", width=1)))
        fore_x = list(range(len(ts_vals), len(ts_vals) + n_periods))
        fig.add_trace(go.Scatter(x=fore_x, y=forecast_vals * 100,
                                 name="Forecast", line=dict(color=DANGER, width=2.5),
                                 mode="lines+markers", marker=dict(size=6)))
        fig.add_hline(y=55, line_dash="dot", line_color=WARN, annotation_text="High-risk threshold (55%)")
        fig.update_layout(title="Maintenance Rate Forecast (Holt-Winters Exponential Smoothing)",
                          xaxis_title="Operational Period", yaxis_title="Maintenance Rate (%)",
                          paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                          font_color="white", title_font_color=ACCENT)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("FORECAST SUMMARY")
        fore_df = pd.DataFrame({
            "Period": [f"+{i+1}" for i in range(n_periods)],
            "Forecast Rate (%)": [f"{v*100:.2f}%" for v in forecast_vals],
            "Risk Level": ["🔴 HIGH" if v > 0.55 else ("🟡 MEDIUM" if v > 0.45 else "🟢 LOW") for v in forecast_vals]
        })
        st.dataframe(fore_df, use_container_width=True)
    except Exception as e:
        st.error(f"Forecasting error: {e}")

# ─────────────────────────────────────────────────────────────
# TAB 5: PUMP REPORT
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("DETAILED PUMP REPORT")
    pump_select = st.selectbox("Select pump", sorted(df["Pump_ID"].unique()), format_func=lambda x: f"Pump {x}")
    pump_df = df[df["Pump_ID"] == pump_select]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Maintenance Rate", f"{pump_df['Maintenance_Flag'].mean()*100:.1f}%")
    c2.metric("Avg Vibration",    f"{pump_df['Vibration'].mean():.2f} mm/s")
    c3.metric("Avg Temperature",  f"{pump_df['Temperature'].mean():.1f} °C")
    c4.metric("Anomalies",        f"{pump_df['is_anomaly'].sum()}")

    # Sensor time series (using index as proxy for time)
    st.subheader(f"SENSOR READINGS — Pump {pump_select}")
    sensor_show = st.multiselect("Sensors to plot", SENSOR_COLS, default=["Temperature", "Vibration"])
    sample_pump = pump_df.sample(min(500, len(pump_df)), random_state=42).sort_values("Operational_Hours")
    fig = make_subplots(rows=len(sensor_show), cols=1, shared_xaxes=True,
                        subplot_titles=sensor_show)
    for i, s in enumerate(sensor_show, 1):
        color = DANGER if sample_pump[s].mean() > df[s].mean() * 1.05 else ACCENT
        fig.add_trace(go.Scatter(x=sample_pump["Operational_Hours"], y=sample_pump[s],
                                 name=s, line=dict(color=color, width=1), opacity=0.8), row=i, col=1)
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
                      font_color="white", height=150 * len(sensor_show) + 100,
                      title_font_color=ACCENT, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Raw data sample
    st.subheader("RAW DATA SAMPLE")
    st.dataframe(pump_df[["Pump_ID"] + SENSOR_COLS + ["Maintenance_Flag", "is_anomaly", "anomaly_raw", "Wear_Index"]].head(100),
                 use_container_width=True)

# ── FOOTER ───────────────────────────────────────────────────
st.markdown("---")
st.caption("PumpSentinel · Isolation Forest + Random Forest + Holt-Winters · Dataset: Large_Industrial_Pump_Maintenance_Dataset.csv · 20,000 records · 5 pumps")
