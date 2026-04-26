import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")

st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)

st.title("📊 Real-Time ML Fraud Detection Dashboard")

LOG_FILE = "logs/predictions.csv"

if not os.path.exists(LOG_FILE):
    import demo_data

try:
    df = pd.read_csv(LOG_FILE)

    total_predictions = len(df)
    total_frauds      = int(df["prediction"].sum())
    fraud_rate        = df["prediction"].mean()

    chart_df = df.tail(1000).reset_index(drop=True)

    st.subheader("📌 Live System Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Total Predictions", f"{total_predictions:,}")
    col2.metric("🚨 Total Frauds Detected", f"{total_frauds:,}")
    col3.metric("📊 Overall Fraud Rate", f"{fraud_rate:.3%}")
    col4.metric("⏱️ Stream Rate", "~3.3 rows/sec")

    st.markdown("---")

    st.subheader("📈 Rolling Fraud Rate — last 1,000 transactions (window=50)")
    chart_df["rolling_fraud_rate"] = chart_df["prediction"].rolling(window=50, min_periods=1).mean()
    st.line_chart(chart_df["rolling_fraud_rate"])

    st.markdown("---")

    if "Amount" in chart_df.columns:
        st.subheader("💰 Transaction Amount — last 1,000 transactions")
        st.line_chart(chart_df["Amount"])

    st.markdown("---")

    st.subheader("🚨 Latest Flagged Fraud Transactions (last 50)")
    fraud_df = df[df["prediction"] == 1].tail(50)
    if fraud_df.empty:
        st.info("No fraud detected yet.")
    else:
        st.dataframe(fraud_df, use_container_width=True)

except Exception as e:
    st.error(f"Dashboard error: {e}")