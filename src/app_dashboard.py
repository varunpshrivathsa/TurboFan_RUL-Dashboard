# src/app_dashboard.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, pathlib, requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import matplotlib as mpl

# ---------------- CONFIG ----------------
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Turbofan RUL Model Evaluation",
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

# ---------------- UTILS ----------------
@st.cache_data
def load_data():
    # Placeholder dummy arrays (Streamlit Cloud doesn't have local .npy files)
    import numpy as np
    X_test = np.zeros((10, 5))  # adjust shape if necessary
    y_test = np.zeros(10)
    return X_test, y_test


@st.cache_resource
def load_models():
    models = {}
    paths = {
        "Linear Regression": "linear_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost (Baseline)": "xgboost.pkl",
        "XGBoost (Tuned)": "xgb_tuned.pkl"
    }
    for name, file in paths.items():
        pth = MODEL_DIR / file
        if pth.exists():
            models[name] = joblib.load(pth)
    return models

def evaluate(model, X, y):
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X = X.copy()
        X[inds] = np.take(col_means, inds[1])

    try:
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)
        return preds, {"RMSE": rmse, "MAE": mae, "R2": r2}
    except Exception as e:
        st.warning(f"{type(model).__name__} failed: {e}")
        return np.zeros_like(y), {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan}


# ---------------- HEADER ----------------
st.markdown("""
    <div style='padding:10px 0;border-bottom:1px solid #333;'>
        <h1 style='color:#00B4D8;font-weight:600;margin-bottom:0;'>Turbofan RUL Analytics Dashboard</h1>
        <p style='color:#888;font-size:14px;margin-top:4px;'>Model Comparison • Diagnostics • Explainability • Live Prediction</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
X_test, y_test = load_data()
models = load_models()

# Evaluate all models
results, preds_dict = {}, {}
for name, model in models.items():
    preds, metrics = evaluate(model, X_test, y_test)
    results[name] = metrics
    preds_dict[name] = preds

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
show_residuals = st.sidebar.checkbox("Show Residual Diagnostics", value=True)
show_importance = st.sidebar.checkbox("Show Feature Importance", value=True)
st.sidebar.markdown("---")
st.sidebar.info("Use toggles to explore model behavior interactively.")

# ---------------- MAIN CONTENT ----------------
tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Predictions", "Explainability", "Live Prediction"])

# --- TAB 1: Metrics
with tab1:
    st.subheader("Model Performance Comparison")
    results_df = pd.DataFrame(results).T
    if "RMSE" in results_df.columns:
        results_df = results_df.sort_values("RMSE")
        st.dataframe(
            results_df.style.format("{:.3f}")
            .highlight_min(color="#90EE90", axis=0)
            .set_table_styles([{
                'selector': 'thead th',
                'props': [('background-color', '#1e1e1e'), ('color', '#00B4D8')]
            }])
        )
    else:
        st.error("Metrics not computed correctly. Check model outputs.")

# --- TAB 2: Predictions and Residuals
with tab2:
    col1, col2 = st.columns(2)
    preds = preds_dict[selected_model]

    # Actual vs Predicted
    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(y_test, preds, alpha=0.45, color="#00B4D8", s=15)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                color="#FF5C5C", linestyle="--", linewidth=1)
        ax.set_xlabel("Actual RUL")
        ax.set_ylabel("Predicted RUL")
        ax.set_title(f"{selected_model} – Actual vs Predicted")
        plt.tight_layout()
        st.pyplot(fig)

    # Error Distribution
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        for name, p in preds_dict.items():
            sns.kdeplot(p - y_test, label=name, fill=True, alpha=0.3, ax=ax2, linewidth=1.3)
        ax2.axvline(0, color="#FF5C5C", linestyle="--", linewidth=1)
        ax2.set_xlabel("Prediction Error (Predicted - Actual)")
        ax2.set_title("Error Distribution")
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

    if show_residuals:
        st.subheader("Residual Diagnostics")
        residuals = preds - y_test
        fig3, ax3 = plt.subplots(1, 2, figsize=(12,4))
        sns.scatterplot(x=preds, y=residuals, ax=ax3[0], alpha=0.4, color="#FFD166", s=15)
        ax3[0].axhline(0, color="#FF5C5C", linestyle="--")
        ax3[0].set_xlabel("Predicted RUL")
        ax3[0].set_ylabel("Residual (Pred - Actual)")
        ax3[0].set_title("Residuals vs Predicted")

        sns.histplot(residuals, kde=True, bins=40, ax=ax3[1], color="#34d399")
        ax3[1].axvline(0, color="#FF5C5C", linestyle="--")
        ax3[1].set_title("Residual Distribution")
        plt.tight_layout()
        st.pyplot(fig3)

# --- TAB 3: Explainability (Feature Importance)
with tab3:
    if show_importance and isinstance(models[selected_model], XGBRegressor):
        st.subheader(f"Feature Importance – {selected_model}")
        importance = models[selected_model].get_booster().get_score(importance_type="gain")
        importance_df = pd.DataFrame(
            sorted(importance.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature","Importance"]
        )
        fig4, ax4 = plt.subplots(figsize=(10,4))
        sns.barplot(data=importance_df.head(15), x="Feature", y="Importance", ax=ax4, palette="cool")
        ax4.set_title("Top 15 Important Features")
        ax4.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig4)
    else:
        st.info("Feature importance is only available for XGBoost models.")

# --- TAB 4: Live API Prediction ---
with tab4:
    st.subheader("Predict RUL using FastAPI Endpoint")
    api_url = st.text_input("Enter API Endpoint", "http://44.204.118.207:8000/predict")

    uploaded_file = st.file_uploader("Upload CSV file with sensor readings", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview", df.head())

        if st.button("Send to API for Prediction"):
            with st.spinner("Sending data to API..."):
                try:
                    res = requests.post(api_url, json={"features": df.values.tolist()})
                    if res.status_code == 200:
                        preds = res.json().get("predictions", [])
                        df["Predicted_RUL"] = preds
                        st.success("Prediction Complete!")
                        st.dataframe(df)
                        st.bar_chart(df["Predicted_RUL"])
                    else:
                        st.error(f"API Error: {res.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")
