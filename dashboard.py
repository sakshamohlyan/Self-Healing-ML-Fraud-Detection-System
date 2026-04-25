import streamlit as st
import pandas as pd
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="ML Monitoring Dashboard", layout="wide")


st_autorefresh(interval=5000, key="dashboard_refresh")  # every 5 seconds

st.title("📊 Real-Time ML Fraud Detection Dashboard")

try:

    df = pd.read_csv("logs/predictions.csv")

    total_predictions = len(df)
    total_frauds      = int(df["prediction"].sum())
    fraud_rate        = df["prediction"].mean()


    rows_per_sec = total_predictions / max(total_predictions / 3.3, 1)


    chart_df = df.tail(1000).reset_index(drop=True)

    # ── METRICS ───────────────────────────────────────────────────────────────
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
        value=f"~{rows_per_sec:.1f} rows/sec",
        help="Calculated from total rows processed."
    )

    st.markdown("---")

    # ── ROLLING FRAUD RATE CHART ──────────────────────────────────────────────

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

    # ── TRANSACTION AMOUNT CHART ──────────────────────────────────────────────
    if "Amount" in chart_df.columns:
        st.subheader("💰 Transaction Amount — last 1,000 transactions")
        st.caption(
            "Sharp spikes = injected anomalies (Amount × 5–15x). "
            "Correlate with fraud flags in the table below."
        )
        st.line_chart(chart_df["Amount"])

    st.markdown("---")

    # ── FRAUD TABLE ───────────────────────────────────────────────────────────
    st.subheader("🚨 Latest Flagged Fraud Transactions (last 50)")
    st.caption("Rows where prediction=1. High Amount + distorted V1/V2/V3 = injected anomaly.")
    fraud_df = df[df["prediction"] == 1].tail(50)
    if fraud_df.empty:
        st.info("No fraud detected yet.")
    else:
        st.dataframe(fraud_df, use_container_width=True)

except FileNotFoundError:
    st.warning("⚠️ No log file yet — waiting for the pipeline to start.")
except Exception as e:
    st.error(f"Dashboard error: {e}")