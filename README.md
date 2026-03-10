# Traffic Volume Prediction (ML + Streamlit)

A machine learning web application that predicts highway traffic volume based on weather and time conditions.

## Features
- Predict traffic volume using machine learning
- Convert predictions into traffic levels (Low / Moderate / Heavy)
- Interactive Streamlit web interface
- Uses real-world traffic and weather dataset

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn

## How it Works
1. Load and clean the traffic dataset
2. Extract time-based features from timestamps
3. Train a Random Forest regression model
4. Predict traffic volume using weather and time inputs
5. Convert predicted vehicle count into traffic levels
6. Display results through a Streamlit web interface

## Dataset
Metro Interstate Traffic Volume Dataset (UCI Machine Learning Repository)

Features used:
- temp
- rain_1h
- snow_1h
- clouds_all
- hour
- day
- month
- day_of_week

Target variable:
traffic_volume

## Model Performance
MAE ≈ 132 vehicles  
R² ≈ 0.38

## Run the App

Install dependencies:

pip install -r requirements.txt

Start the app:

streamlit run app.py