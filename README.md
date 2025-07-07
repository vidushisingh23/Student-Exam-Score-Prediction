# Student Final Grade Predictor App

https://student-exam-score-prediction-vidushi.streamlit.app
An interactive machine learning web application built using **Streamlit** that predicts student final exam scores (G3) based on academic indicators such as G1, G2, study time, and absences. The app includes prediction,data visualizations, and an embedded Power BI dashboard for additional insight.

---

## Features

- Predict final exam grade (G3) using key academic features
- Exploratory data analysis and visualizations
- Model evaluation metrics (R² Score, MAE, RMSE)-done in EDA_and_modeling file while training
- Power BI dashboard summary for the Math dataset
- Deployed on Streamlit Cloud: [Visit App](https://student-exam-score-prediction-vidushi.streamlit.app/)

---

## How to Run the App Locally

### 1. Clone the repository

```bash
git clone[ https://github.com/yourusername/student-score-predictor.git](https://github.com/vidushisingh23/Student-Exam-Score-Prediction)
cd student-exam-score-predictor
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # For Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run streamlit_app.py
```

---

## Project Structure

```
student-score-predictor/
│
├── EDA_and_Modeling.py           # Script for analysis and model training 
├── requirements.txt              # Python dependency list
│
├── model/
│   ├── regressor.pkl             # Trained regression model
│   ├── scaler.pkl                # Scaler used for normalization
│   └── metrics.pkl               # Stored evaluation metrics
│
├── data/
│   ├── math_dataset.csv          # Math subject dataset
│   ├── portugese_dataset.csv     # Portuguese subject dataset
│   └── plots/
│       ├── top10_corr_g3.png
│       ├── studytime_vs_g3.png
│       ├── absences_vs_g3.png
│       ├── actual_vs_pred.png
│       ├── eda_heatmap.png
│       └── powerbi_dashboard.png
│
└── apps/
    └── streamlit_app.py         # Main Streamlit interface
```



---

## Author

**Vidushi Singh**  
– Student Exam Score Prediction

---
