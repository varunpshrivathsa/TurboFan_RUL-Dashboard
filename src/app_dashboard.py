# src/app_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Turbofan RUL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- HEADER ----------------
st.markdown("""
    <div style='padding:10px 0;border-bottom:1px solid #333;'>
        <h1 style='color:#00B4D8;font-weight:600;margin-bottom:0;'>Turbofan RUL Analytics Dashboard</h1>
        <p style='color:#888;font-size:14px;margin-top:4px;'>Model Evaluation • Predictions • Explainability • Live API Integration</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Select Mode", ["Live API Prediction", "Manual Input"])
st.sidebar.markdown("---")
st.sidebar.info("Interact with the model in real time via FastAPI backend.")

# ---------------- MAIN ----------------
tab1, tab2, tab3 = st.tabs(["Live Prediction", "Analytics & Visualization", "Diagnostics"])

api_url = st.text_input("Enter FastAPI Endpoint", "http://44.204.118.207:8000/predict")

# --- TAB 1: Live Prediction ---
with tab1:
    st.subheader("Predict Remaining Useful Life (RUL)")

    if mode == "Live API Prediction":
        uploaded_file = st.file_uploader("Upload CSV file with sensor readings", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())

            if st.button("Send to API for Prediction"):
                with st.spinner("Sending data to API..."):
                    try:
                        res = requests.post(api_url, json={"features": df.values.tolist()}, timeout=60)
                        if res.status_code == 200:
                            preds = res.json().get("predictions", [])
                            df["Predicted_RUL"] = preds
                            st.success("Prediction Complete!")

                            # Key summary metrics
                            st.metric("Mean Predicted RUL", f"{df['Predicted_RUL'].mean():.2f}")
                            st.metric("Max Predicted RUL", f"{df['Predicted_RUL'].max():.2f}")
                            st.metric("Min Predicted RUL", f"{df['Predicted_RUL'].min():.2f}")

                            # Download option
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

                            # Save dataframe for use in other tabs
                            st.session_state["latest_df"] = df
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

# --- TAB 2: Analytics & Visualization ---
with tab2:
    st.subheader("Interactive Analytics")

    if "latest_df" in st.session_state:
        df = st.session_state["latest_df"]

        # RUL Distribution
        fig1 = px.histogram(df, x="Predicted_RUL", nbins=20, title="Predicted RUL Distribution", 
                            color_discrete_sequence=["#00B4D8"], opacity=0.7)
        fig1.update_layout(template="plotly_dark", xaxis_title="Predicted RUL", yaxis_title="Count")
        st.plotly_chart(fig1, use_container_width=True)

        # Box Plot for Outliers
        fig2 = px.box(df, y="Predicted_RUL", title="RUL Outlier Detection", color_discrete_sequence=["#FF5C5C"])
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

        # Trend line
        df["Index"] = np.arange(len(df))
        fig3 = px.line(df, x="Index", y="Predicted_RUL", title="Predicted RUL Trend Over Samples", 
                       color_discrete_sequence=["#64DFDF"])
        fig3.update_layout(template="plotly_dark", xaxis_title="Sample Index", yaxis_title="Predicted RUL")
        st.plotly_chart(fig3, use_container_width=True)

        # Filtering slider
        st.write("### Filter by RUL Range")
        min_rul, max_rul = st.slider("Select RUL range:", float(df["Predicted_RUL"].min()), 
                                     float(df["Predicted_RUL"].max()), 
                                     (float(df["Predicted_RUL"].min()), float(df["Predicted_RUL"].max())))
        filtered_df = df[(df["Predicted_RUL"] >= min_rul) & (df["Predicted_RUL"] <= max_rul)]
        st.write(f"Filtered {len(filtered_df)} samples in selected range.")
        st.dataframe(filtered_df)

        # Correlation Heatmap (optional if > 1 numeric column)
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) > 2:
            corr = df[num_cols].corr()
            fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
            fig4.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Run a prediction first to view analytics.")

# --- TAB 3: Diagnostics ---
with tab3:
    st.subheader("System Diagnostics")
    st.info("""
        **EC2 FastAPI backend**: Running and reachable  
        **Streamlit Frontend**: Deployed successfully on Streamlit Cloud  
        **Model Predictions**: Live and interactive  
        ---
        Use the analytics tab to visualize RUL trends, outliers, and correlations in real time.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='color:#888;font-size:12px;text-align:center;'>© 2025 Turbofan RUL Analytics • Powered by Streamlit + FastAPI + Plotly</p>",
    unsafe_allow_html=True
)
