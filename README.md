
---

### For the Dashboard Repo (`TurboFan_RUL-Dashboard`)
Copy the following into its own `README.md`:

```markdown
# TurboFan_RUL-Dashboard  
Interactive Streamlit dashboard for visualizing Remaining Useful Life (RUL) predictions and model performance metrics from the TurboFanRUL backend API.

---

## Features
- Upload or select sample engine data
- Send data to FastAPI backend for prediction
- Visualize actual vs predicted RUL
- Explore feature trends and error analysis
- Multiple plot types (Plotly, Matplotlib)
- Modern dark UI styling

---

## Project Structure
TurboFan_RUL-Dashboard/
│
├── src/
│ ├── app_dashboard.py
│ └── utils.py
├── data/
│ ├── sample_valid_input.csv
├── requirements.txt
└── README.md


---

##  Setup & Run Locally

### 1 Clone the repository
```bash
git clone https://github.com/varunpshrivathsa/TurboFan_RUL-Dashboard.git
cd TurboFan_RUL-Dashboard

### 2 Install Dependencies
pip install -r requirements.txt

### 3 Run Streamlit app
streamlit run src/app_dashboard.py
Then open http://localhost:8501 in your browser.

### 4 Connect to Backend API
By default, the dashboard sends requests to:
API_URL = "http://<EC2-IP>:8000/predict"
Edit this in app_dashboard.py if you deploy elsewhere.

Backend API
This dashboard connects to the main backend:
TurbofanRUL

MIT License © 2025 Varun Phanindra Shrivathsa