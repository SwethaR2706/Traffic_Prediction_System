import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/traffic_model.pkl")

st.title("Traffic Volume Prediction")

st.write("Enter weather and time details to estimate traffic volume.")

temp = st.number_input("Temperature (Kelvin)", value=280.0)
rain = st.number_input("Rain in last hour (mm)", value=0.0)
snow = st.number_input("Snow in last hour (mm)", value=0.0)
clouds = st.slider("Cloud coverage (%)", 0, 100, 50)

hour = st.slider("Hour of the day", 0, 23, 12)
day = st.slider("Day of the month", 1, 31, 15)
month = st.slider("Month", 1, 12, 6)
day_of_week = st.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 3)

if st.button("Predict Traffic"):

    features = np.array([[temp, rain, snow, clouds, hour, day, month, day_of_week]])

    prediction = model.predict(features)
    volume = int(prediction[0])

    if volume < 1500:
        level = "Low Traffic"
    elif volume < 4000:
        level = "Moderate Traffic"
    else:
        level = "Heavy Traffic"

    st.write(f"Estimated Vehicle Count: {volume}")
    st.success(f"Traffic Level: {level}")