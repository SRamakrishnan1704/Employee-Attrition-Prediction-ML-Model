import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Define app directory
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# === STREAMLIT APP ===

st.set_page_config(page_title="Employee Prediction Dashboard", layout="wide")
st.title("🔮 Employee Prediction Dashboard")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    attrition_model = joblib.load(os.path.join(APP_DIR, 'attrition_pipeline.pkl'))
    perf_model = joblib.load(os.path.join(APP_DIR, 'performance_pipeline.pkl'))
    features = joblib.load(os.path.join(APP_DIR, 'features.pkl'))
    return attrition_model, perf_model, features

attrition_model, perf_model, features = load_models()
st.success(f"✅ Models loaded! Uses {len(features)} features")

# Load original data for rankings
@st.cache_data
def load_employee_data():
    data_path = "D:/Ramakrishnan S/Guvi/Visual studio/My Project foler/Employee_Prediction/data/Employee_Attrition.csv"
    df = pd.read_csv(data_path)
    return df

df = load_employee_data()

# === INPUT SECTION ===
st.subheader("📝 Employee Input")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Categorical Features**")
    business_travel = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("MaritalStatus", ["Married", "Single", "Divorced"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])

with col2:
    env_sat = st.slider("Environment Satisfaction", 1, 4, 3)
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)
    job_inv = st.slider("Job Involvement", 1, 4, 3)
    work_life = st.slider("Work Life Balance", 1, 4, 3)

col1, col2, col3 = st.columns(3)
with col1:
    monthly_income = st.number_input("Monthly Income", value=5000, min_value=1000, step=100)
with col2:
    distance_home = st.slider("Distance From Home", 1, 30, 10)
with col3:
    stock_opt = st.slider("Stock Option Level", 0, 3, 1)

col1, col2, col3 = st.columns(3)
with col1:
    num_companies = st.slider("Num Companies Worked", 0, 9, 2)
with col2:
    total_years = st.slider("Total Working Years", 0, 40, 10)
with col3:
    years_company = st.slider("Years At Company", 0, 40, 5)

years_role = st.slider("Years In Current Role", 0, 20, 4)

# === PREDICTION ===
if st.button("🔮 PREDICT", type="primary"):
    # Create input DataFrame (EXACT feature order)
    input_data = pd.DataFrame({
        features[0]: [business_travel],
        features[1]: [distance_home],
        features[2]: [env_sat],
        features[3]: [gender],
        features[4]: [job_inv],
        features[5]: [job_sat],
        features[6]: [marital],
        features[7]: [monthly_income],
        features[8]: [num_companies],
        features[9]: [overtime],
        features[10]: [stock_opt],
        features[11]: [total_years],
        features[12]: [work_life],
        features[13]: [years_company],
        features[14]: [years_role]
    })
    
    # Both predictions
    attrition_pred = attrition_model.predict(input_data)[0]
    attrition_proba = attrition_model.predict_proba(input_data)[0]
    perf_pred = round(perf_model.predict(input_data)[0])
    
    # === RESULTS ===
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Attrition Prediction")
        st.metric("Will Leave?", "Yes" if attrition_pred == 1 else "No", 
                 f"{max(attrition_proba)*100:.1f}%")
        st.metric("Leave Probability", f"{attrition_proba[1]*100:.1f}%")
        st.metric("Stay Probability", f"{attrition_proba[0]*100:.1f}%")
    
    with col2:
        st.markdown("### 🏆 Performance Prediction")
        st.metric("Performance Rating", f"{perf_pred}/4")
        st.success("⭐ Top Performer!" if perf_pred == 4 else "⭐ Good Performer")
    
    st.dataframe(input_data.T, use_container_width=True)

# === TOP/BOTTOM PERFORMERS ===
st.markdown("---")
st.subheader("🏆 Employee Performance Rankings")

# Prepare data for ranking (same preprocessing)
df_fe = df[features + ['PerformanceRating']].copy()
perf_probs = perf_model.predict(df_fe[features])

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔥 Top 10 Performers")
    top_performers = df[['EmployeeNumber', 'PerformanceRating', 'MonthlyIncome', 'JobRole']].head(10)
    st.dataframe(top_performers, use_container_width=True)

with col2:
    st.markdown("### 📉 Bottom 10 Performers (Risk)")
    bottom_performers = df[['EmployeeNumber', 'PerformanceRating', 'MonthlyIncome', 'JobRole']].tail(10)
    st.dataframe(bottom_performers, use_container_width=True)

# Performance distribution chart
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📊 Performance Distribution")
    st.bar_chart(df['PerformanceRating'].value_counts())

st.markdown("---")
st.caption("🎉 Powered by Random Forest + Linear Regression pipelines")