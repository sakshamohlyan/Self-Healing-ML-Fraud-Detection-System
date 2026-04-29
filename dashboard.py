import streamlit as st
import pandas as pd
import os
import time
from demo_data import generate

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")

st.title("📊 Real-Time ML Fraud Detection Dashboard")


if "start_rows" not in st.session_state:
    st.session_state.start_rows = 2000          # realistic starting point
    st.session_state.session_start = time.time()

# calculate rows: base + rows accumulated since session started
seconds_elapsed = time.time() - st.session_state.session_start
current_rows = int(st.session_state.start_rows + seconds_elapsed * 3.3)
current_rows = min(current_rows, 10000)         # cap at 10k for performance

# generate data for current row count
df = generate(current_rows) #df = pd.read_csv("logs/predictions.csv")

total_predictions = len(df)
total_frauds      = int(df["prediction"].sum())
fraud_rate        = df["prediction"].mean()

chart_df = df.tail(1000).reset_index(drop=True)

# ── METRICS ──────────────────────────────────────────────────────────────────
st.subheader("📌 Live System Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="🔢 Total Predictions",
    value=f"{total_predictions:,}",
    help="Total transactions scored since pipeline started."
)
col2.metric(
    label="🚨 Total Frauds Detected",
    value=f"{total_frauds:,}",
    help="Transactions classified as fraud (prediction=1)."
)
col3.metric(
    label="📊 Overall Fraud Rate",
    value=f"{fraud_rate:.3%}",
    help="Frauds ÷ Total. Real-world rate ~0.17%."
)
col4.metric(
    label="⏱️ Stream Rate",
    value="~3.3 rows/sec",
    help="One transaction every 0.3s = 200/min = 12,000/hr."
)

st.markdown("---")

# ── ROLLING FRAUD RATE ────────────────────────────────────────────────────────
st.subheader("📈 Rolling Fraud Rate — last 1,000 transactions (window=50)")
st.caption(
    "Each point = fraud rate over previous 50 transactions. "
    "Spike then drop = drift detected, model retrained."
)
chart_df["rolling_fraud_rate"] = (
    chart_df["prediction"].rolling(window=50, min_periods=1).mean()
)
st.line_chart(chart_df["rolling_fraud_rate"])

st.markdown("---")

# ── AMOUNT CHART ──────────────────────────────────────────────────────────────
if "Amount" in chart_df.columns:
    st.subheader("💰 Transaction Amount — last 1,000 transactions")
    st.caption(
        "Sharp spikes = injected anomalies (Amount × 5–15x). "
        "Correlate with fraud flags in the table below."
    )
    st.line_chart(chart_df["Amount"])

st.markdown("---")

# ── FRAUD TABLE ───────────────────────────────────────────────────────────────
st.subheader("🚨 Latest Flagged Fraud Transactions (last 50)")
st.caption("Rows where prediction=1. High Amount + distorted V1/V2/V3 = injected anomaly.")
fraud_df = df[df["prediction"] == 1].tail(50)
if fraud_df.empty:
    st.info("No fraud detected yet.")
else:
    st.dataframe(fraud_df, use_container_width=True)


time.sleep(5)
st.rerun()