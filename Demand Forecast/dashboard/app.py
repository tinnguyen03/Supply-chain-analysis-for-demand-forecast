"""Small Streamlit viewer for pipeline outputs."""
from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="M5 Demand Forecasting", layout="wide")
st.title("Retail Demand Forecasting & Inventory Simulation")
st.caption("M5 dataset • Baseline experiment viewer")

metrics_path = OUTPUTS / "evaluation_metrics.json"
forecast_path = OUTPUTS / "forecast.csv"
inventory_path = OUTPUTS / "inventory_simulation.csv"

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    cols = st.columns(min(3, len(metrics)))
    for col, (name, value) in zip(cols, metrics.items()):
        if isinstance(value, (int, float)):
            col.metric(name, f"{value:,.3f}" if isinstance(value, float) else f"{value:,}")
else:
    st.info("No evaluation metrics yet. Run `python run_pipeline.py` after adding the M5 files.")

if forecast_path.exists():
    df = pd.read_csv(forecast_path)
    st.subheader("Actual vs predicted demand")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        chart_cols = [c for c in ["sales", "prediction"] if c in df.columns]
        if chart_cols:
            daily = df.groupby("date")[chart_cols].sum()
            st.line_chart(daily)
    st.subheader("Forecast output sample")
    st.dataframe(df.head(100), use_container_width=True)

if inventory_path.exists():
    inv = pd.read_csv(inventory_path)
    st.subheader("Inventory simulation")
    st.dataframe(inv.head(100), use_container_width=True)
