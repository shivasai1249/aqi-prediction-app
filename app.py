import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌍", layout="wide")

model = joblib.load("xgboost_aqi_model.pkl")

st.sidebar.title("About Project")

st.sidebar.info(
"""
Air Quality Prediction System

Model Used:
XGBoost Regressor

Input Features:
PM2.5
PM10
NO2
SO2
CO
O3

Output:
Predicted Air Quality Index (AQI)
"""
)

page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}
[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}
.title{
text-align:center;
font-size:45px;
font-weight:bold;
}
.subtitle{
text-align:center;
font-size:20px;
margin-bottom:30px;
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

st.markdown('<p class="title">🌍 Air Quality Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict AQI using XGBoost Machine Learning Model</p>', unsafe_allow_html=True)

st.markdown("### AQI Categories")

st.table({
"AQI Range": ["0-50","51-100","101-200","201+"],
"Category": ["Good 🟢","Moderate 🟡","Poor 🟠","Hazardous 🔴"]
})

st.markdown("### Enter Pollution Levels")

col1, col2 = st.columns(2)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)
    no2 = st.number_input("NO2", min_value=0.0)

with col2:
    so2 = st.number_input("SO2", min_value=0.0)
    co = st.number_input("CO", min_value=0.0)
    o3 = st.number_input("O3", min_value=0.0)

if st.button("Predict AQI"):

    data = np.array([[pm25, pm10, no2, so2, co, o3]])

    prediction = model.predict(data)[0]

    st.subheader(f"Predicted AQI: {prediction:.2f}")

    if prediction <= 50:
        st.success("Air Quality: Good 🟢")
    elif prediction <= 100:
        st.info("Air Quality: Moderate 🟡")
    elif prediction <= 200:
        st.warning("Air Quality: Poor 🟠")
    else:
        st.error("Air Quality: Hazardous 🔴")

st.markdown("### Model Feature Importance")

fig, ax = plt.subplots()
plot_importance(model, ax=ax)
st.pyplot(fig)

st.markdown("---")
st.markdown("Developed for Machine Learning Project | Air Quality Prediction using XGBoost")