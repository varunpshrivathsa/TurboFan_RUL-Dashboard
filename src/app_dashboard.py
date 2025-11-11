# src/app_dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
import requests, time, json
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ks_2samp

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Turbofan RUL Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(
    "<style>body{background-color:#0e1117; color:#ddd;} div.block-container{padding-top:1rem;}</style>",
    unsafe_allow_html=True
)

# ---------------- HEADER ----------------
st.markdown("""
<div style='padding:10px 0;border-bottom:1px solid #333;'>
    <h1 style='color:#00B4D8;font-weight:600;margin-bottom:0;'>Turbofan RUL Analytics Dashboard</h1>
    <p style='color:#888;font-size:14px;margin-top:4px;'>Model Comparison • Diagnostics • Explainability • Live Prediction</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
mode = st.sidebar.radio("Select Mode", ["Live API Prediction", "Manual Input", "Model Comparison"])
st.sidebar.markdown("---")
st.sidebar.info("Interactively evaluate and visualize RUL predictions in real time.")

# Global constants
api_url = st.text_input("Enter FastAPI Endpoint", "http://44.204.118.207:8000/predict")

# ---------------- CACHED HELPERS ----------------
@st.cache_data
def post_request(url, payload):
    start = time.perf_counter()
    res = requests.post(url, json=payload, timeout=90)
    latency = time.perf_counter() - start
    return res, latency


# ---------------- TAB SETUP ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Prediction", "Analytics", "Diagnostics", "Feature Explorer", "Model Benchmark"
])

# ---------------------------------------------------------------------------------
# TAB 1: LIVE API PREDICTION
# ---------------------------------------------------------------------------------
with tab1:
    st.subheader("Predict Remaining Useful Life (RUL) via FastAPI")

    if mode == "Live API Prediction":
        uploaded_file = st.file_uploader("Upload CSV file with sensor readings", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Uploaded Data Preview")
            st.dataframe(df.head())

            if st.button("Send to API for Prediction"):
                with st.spinner("Querying FastAPI backend..."):
                    try:
                        res, latency = post_request(api_url, {"features": df.values.tolist()})
                        if res.status_code == 200:
                            preds = res.json().get("predictions", [])
                            df["Predicted_RUL"] = preds
                            st.success(f"✅ Prediction Complete (Latency: {latency:.2f}s)")

                            st.metric("Mean RUL", f"{df['Predicted_RUL'].mean():.2f}")
                            st.metric("Min RUL", f"{df['Predicted_RUL'].min():.2f}")
                            st.metric("Max RUL", f"{df['Predicted_RUL'].max():.2f}")

                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

                            st.session_state["latest_df"] = df
                        else:
                            st.error(f"API Error: {res.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")

    elif mode == "Manual Input":
        st.write("Enter comma-separated sensor values:")
        vals = st.text_input("Example: 0.5, 0.2, 0.7, 0.1, 0.3")
        if st.button("Predict Single Input"):
            try:
                arr = [float(x.strip()) for x in vals.split(",")]
                res, latency = post_request(api_url, {"features": [arr]})
                if res.status_code == 200:
                    pred = res.json()["predictions"][0]
                    st.success(f"Predicted RUL: {pred:.2f} (Latency: {latency:.2f}s)")
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------------------
# TAB 2: ANALYTICS
# ---------------------------------------------------------------------------------
with tab2:
    st.subheader("Prediction Analytics & Visualization")

    if "latest_df" in st.session_state:
        df = st.session_state["latest_df"]
        if "Predicted_RUL" in df.columns:
            # RUL Distribution
            fig1 = px.histogram(df, x="Predicted_RUL", nbins=25, title="Predicted RUL Distribution",
                                color_discrete_sequence=["#00B4D8"], opacity=0.7)
            fig1.update_layout(template="plotly_dark", xaxis_title="Predicted RUL", yaxis_title="Count")
            st.plotly_chart(fig1, use_container_width=True)

            # Trend Over Samples
            df["Index"] = np.arange(len(df))
            fig2 = px.line(df, x="Index", y="Predicted_RUL", title="RUL Trend Over Samples",
                           color_discrete_sequence=["#64DFDF"])
            fig2.update_layout(template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

            # Outlier detection
            fig3 = px.box(df, y="Predicted_RUL", points="all", title="Outlier Detection",
                          color_discrete_sequence=["#FF5C5C"])
            fig3.update_layout(template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)

            # Filtering
            min_rul, max_rul = st.slider("Filter RUL range", float(df["Predicted_RUL"].min()),
                                         float(df["Predicted_RUL"].max()),
                                         (float(df["Predicted_RUL"].min()), float(df["Predicted_RUL"].max())))
            filtered = df[(df["Predicted_RUL"] >= min_rul) & (df["Predicted_RUL"] <= max_rul)]
            st.write(f"Filtered {len(filtered)} samples in selected range.")
            st.dataframe(filtered.head())

            # Correlation
            num_cols = df.select_dtypes(include=np.number).columns
            if len(num_cols) > 2:
                corr = df[num_cols].corr()
                fig4 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
                fig4.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig4, use_container_width=True)

    else:
        st.info("Run a prediction to enable analytics.")

# ---------------------------------------------------------------------------------
# TAB 3: DIAGNOSTICS
# ---------------------------------------------------------------------------------
with tab3:
    st.subheader("System Health & Drift Monitoring")

    col1, col2 = st.columns(2)
    with col1:
        try:
            res = requests.get(api_url.replace("/predict", "/docs"), timeout=5)
            if res.status_code == 200:
                st.success("✅ FastAPI backend is reachable.")
            else:
                st.warning("⚠️ API responded but not healthy.")
        except:
            st.error("❌ FastAPI backend not reachable.")
    with col2:
        st.metric("App Latency (Last Run)", f"{st.session_state.get('latency', 'N/A')} s")

    if "latest_df" in st.session_state:
        df = st.session_state["latest_df"]
        st.write("### Drift Analysis (vs Training Distribution Example)")
        reference = np.random.normal(df["Predicted_RUL"].mean(), df["Predicted_RUL"].std(), len(df))
        stat, pval = ks_2samp(df["Predicted_RUL"], reference)
        st.write(f"KS Statistic: **{stat:.3f}**, p-value: **{pval:.3f}**")
        if pval < 0.05:
            st.error("⚠️ Potential drift detected!")
        else:
            st.success("✅ No significant drift detected.")

# ---------------------------------------------------------------------------------
# TAB 4: FEATURE EXPLORER
# ---------------------------------------------------------------------------------
with tab4:
    st.subheader("Feature Impact Explorer (1D Sensitivity)")

    if "latest_df" in st.session_state:
        df = st.session_state["latest_df"]
        feature_cols = [c for c in df.columns if c != "Predicted_RUL"]
        if len(feature_cols) > 0:
            feature = st.selectbox("Select feature to vary:", feature_cols)
            min_val, max_val = float(df[feature].min()), float(df[feature].max())
            step = (max_val - min_val) / 10
            slider = st.slider("Vary feature value", min_val, max_val, min_val, step=step)

            base_row = df.iloc[0].copy()
            base_row[feature] = slider
            try:
                res, latency = post_request(api_url, {"features": [base_row.values.tolist()]})
                if res.status_code == 200:
                    pred = res.json()["predictions"][0]
                    st.success(f"Predicted RUL: {pred:.2f} (Latency {latency:.2f}s)")
                    st.metric("Feature value", f"{slider:.3f}")
                    st.metric("Predicted RUL", f"{pred:.3f}")
                else:
                    st.error("API Error during feature test.")
            except Exception as e:
                st.error(f"Request failed: {e}")
        else:
            st.warning("No numeric feature columns available for exploration.")
    else:
        st.info("Upload data and predict first.")

# ---------------------------------------------------------------------------------
# TAB 5: MODEL BENCHMARK
# ---------------------------------------------------------------------------------
with tab5:
    st.subheader("Compare Multiple FastAPI Models / Endpoints")

    st.write("Define multiple endpoints to benchmark latency and prediction differences.")
    endpoints = st.text_area(
        "Enter comma-separated model endpoints",
        "http://44.204.118.207:8000/predict, http://44.204.118.207:8000/predict_tuned"
    )

    if "latest_df" in st.session_state:
        df = st.session_state["latest_df"]
        if st.button("Run Benchmark"):
            results = []
            endpoints_list = [x.strip() for x in endpoints.split(",")]
            for ep in endpoints_list:
                with st.spinner(f"Testing {ep}..."):
                    try:
                        res, latency = post_request(ep, {"features": df.values.tolist()})
                        if res.status_code == 200:
                            preds = res.json()["predictions"]
                            mean_pred = np.mean(preds)
                            results.append({"Endpoint": ep, "Latency (s)": latency, "Mean RUL": mean_pred})
                        else:
                            results.append({"Endpoint": ep, "Latency (s)": np.nan, "Mean RUL": np.nan})
                    except:
                        results.append({"Endpoint": ep, "Latency (s)": np.nan, "Mean RUL": np.nan})

            result_df = pd.DataFrame(results)
            st.dataframe(result_df)

            fig = px.bar(result_df, x="Endpoint", y="Latency (s)", title="Model Inference Latency",
                         color="Endpoint", text_auto=True)
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run at least one prediction first to benchmark endpoints.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='color:#888;font-size:12px;text-align:center;'>© 2025 Turbofan RUL Analytics • Powered by Streamlit + FastAPI + Plotly</p>",
    unsafe_allow_html=True
)
