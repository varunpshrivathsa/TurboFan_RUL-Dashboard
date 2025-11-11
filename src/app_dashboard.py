# src/app_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Turbofan RUL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- STYLE ----------------
mpl.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#ddd",
    "xtick.color": "#aaa",
    "ytick.color": "#aaa",
    "grid.color": "#333",
    "axes.titleweight": "semibold",
    "axes.titlecolor": "#00B4D8",
    "axes.titlesize": 13,
    "font.size": 11,
    "legend.facecolor": "#0e1117",
    "legend.edgecolor": "#222",
    "text.color": "#ddd",
})
sns.set_theme(style="darkgrid")

# ---------------- HEADER ----------------
st.markdown("""
    <div style='padding:10px 0;border-bottom:1px solid #333;'>
        <h1 style='color:#00B4D8;font-weight:600;margin-bottom:0;'>Turbofan RUL Analytics Dashboard</h1>
        <p style='color:#888;font-size:14px;margin-top:4px;'>Model Comparison • Diagnostics • Explainability • Live Prediction</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Select Mode", ["Live API Prediction", "Manual Input"])
st.sidebar.markdown("---")
st.sidebar.info("Use toggles to explore model predictions interactively.")

# ---------------- MAIN CONTENT ----------------
tab1, tab2 = st.tabs(["Live Prediction", "Diagnostics"])

# --- TAB 1: Live API Prediction ---
with tab1:
    st.subheader("Predict Remaining Useful Life (RUL) using FastAPI Endpoint")

    api_url = st.text_input("Enter FastAPI Endpoint", "http://44.204.118.207:8000/predict")

    if mode == "Live API Prediction":
        uploaded_file = st.file_uploader("Upload CSV file with sensor readings", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview", df.head())

            if st.button("Send to API for Prediction"):
                with st.spinner("Sending data to API..."):
                    try:
                        res = requests.post(api_url, json={"features": df.values.tolist()}, timeout=60)
                        if res.status_code == 200:
                            preds = res.json().get("predictions", [])
                            df["Predicted_RUL"] = preds
                            st.success("Prediction Complete!")
                            st.dataframe(df)

                            st.write("### RUL Distribution")
                            fig, ax = plt.subplots(figsize=(8, 4))
                            sns.histplot(df["Predicted_RUL"], bins=20, kde=True, ax=ax, color="#00B4D8")
                            ax.set_xlabel("Predicted RUL")
                            ax.set_ylabel("Frequency")
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.error(f"API Error ({res.status_code}): {res.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    elif mode == "Manual Input":
        st.write("Enter comma-separated sensor values:")
        vals = st.text_input("Example: 0.5, 0.2, 0.7, 0.1, 0.3")
        if st.button("Predict Single Input"):
            try:
                arr = [float(x.strip()) for x in vals.split(",")]
                res = requests.post(api_url, json={"features": [arr]}, timeout=30)
                if res.status_code == 200:
                    pred = res.json()["predictions"][0]
                    st.success(f"Predicted RUL: {pred:.2f}")
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# --- TAB 2: Diagnostics (optional visuals placeholder)
with tab2:
    st.subheader("System Diagnostics")
    st.info("""
        This dashboard version focuses on **real-time API predictions**.  
        Model evaluation, comparison, and explainability are computed server-side (FastAPI backend).
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='color:#888;font-size:12px;text-align:center;'>© 2025 Turbofan RUL Analytics • Powered by Streamlit + FastAPI</p>",
    unsafe_allow_html=True
)
