import joblib
import pandas as pd
import streamlit as st

FEATURES = ["temp", "rain_1h", "snow_1h", "clouds_all",
            "hour", "day", "month", "day_of_week", "is_holiday"]
LOW, HIGH = 1500, 4000

st.set_page_config(page_title="Traffic Volume Prediction")


@st.cache_resource
def load_model():
    return joblib.load("model/traffic_model.pkl")


model = load_model()

st.title("Traffic Volume Prediction")
st.write("Estimate hourly traffic volume on a US interstate from weather and time of day.")

col1, col2 = st.columns(2)
with col1:
    temp_c = st.number_input("Temperature (°C)", value=10.0, min_value=-40.0, max_value=45.0)
    rain = st.number_input("Rain in last hour (mm)", value=0.0, min_value=0.0)
    snow = st.number_input("Snow in last hour (mm)", value=0.0, min_value=0.0)
    clouds = st.slider("Cloud coverage (%)", 0, 100, 50)
with col2:
    hour = st.slider("Hour of the day", 0, 23, 8)
    day = st.slider("Day of the month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    dow_name = st.selectbox("Day of week",
                            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    holiday = st.checkbox("Public holiday")

if st.button("Predict Traffic"):
    row = pd.DataFrame([{
        "temp": temp_c + 273.15,          # the dataset stores Kelvin
        "rain_1h": rain, "snow_1h": snow, "clouds_all": clouds,
        "hour": hour, "day": day, "month": month,
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday",
                        "Friday", "Saturday", "Sunday"].index(dow_name),
        "is_holiday": int(holiday),
    }])[FEATURES]
    volume = int(model.predict(row)[0])

    if volume < LOW:
        level, show = "Low Traffic", st.success
    elif volume < HIGH:
        level, show = "Moderate Traffic", st.warning
    else:
        level, show = "Heavy Traffic", st.error

    st.metric("Estimated vehicles per hour", f"{volume:,}")
    show(f"Traffic Level: {level}")
