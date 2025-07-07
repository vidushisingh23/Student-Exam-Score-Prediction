import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Student Score Predictor",
    layout="wide",
    page_icon="📊"
)

# Load model and scaler
with open("model/regressor.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# App styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #f5f6fa;
    }
    .stApp {
        background-color: #0e1117;
        color: #f5f6fa;
    }
    h1, h2, h3 {
        color: #f39c12;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Prediction", "📊 Visualizations", "📈Student Performance dashboard"])

# Page 1: Prediction Interface
if page == "🏠 Prediction":
    st.title("🎓 Student Final Grade Predictor")

    st.markdown("Predict a student's final exam grade (G3) based on input values.")

    col1, col2 = st.columns(2)

    with col1:
        G1 = st.slider("First Period Grade (G1)", 0, 20, 10)
        studytime = st.selectbox("Weekly Study Time (hours)", [1, 2, 3, 4])

    with col2:
        G2 = st.slider("Second Period Grade (G2)", 0, 20, 10)
        absences = st.slider("Number of Absences", 0, 50, 5)

    subject = st.radio("Subject", ["Math", "Portuguese"])

    input_df = pd.DataFrame({
        'G1': [G1],
        'G2': [G2],
        'studytime': [studytime],
        'absences': [absences]
    })

    scaled_input = scaler.transform(input_df)

    if st.button("🔍 Predict Grade"):
        prediction = model.predict(scaled_input)[0]
        st.success(f"Predicted Final Grade (G3): {prediction:.2f}")

# Page 2: Visualizations
elif page == "📊 Visualizations":
    st.title("📊 Data Insights")

    st.markdown("Gain insights into student performance trends and correlations.")

    col1, col2 = st.columns(2)

    with col1:
        st.image("data/plots/top10_corr_g3.png", caption="Top 10 Features Correlated with G3", use_container_width=True)
        st.image("data/plots/studytime_vs_g3.png", caption="Study Time vs Final Grade", use_container_width=True)

    with col2:
        st.image("data/plots/absences_vs_g3.png", caption="Absences vs G3", use_container_width=True)
        st.image("data/plots/actual_vs_pred.png", caption="Actual vs Predicted G3", use_container_width=True)

    st.image("data/plots/eda_heatmap.png", caption="Correlation Heatmap of Top Features", use_container_width=True)

# Page 3: power bi dashboard
elif page == "📈Student Performance dashboard":
    st.title("📈Student Performance dashboard")

    st.markdown("""
    <div style='color:lightgray;'>
            (Note: Interactivity is not possible.)
        
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("data/plots/powerbi_dashboard.png"):
        st.image("data/plots/powerbi_dashboard.png", use_container_width=True)
    else:
        st.warning("Power BI dashboard image not found.")
    st.markdown("""
    <div style='color:lightgray;'>
        
    This dashboard is generated using Power BI Desktop based on the Math dataset
    </div>
    """, unsafe_allow_html=True)    