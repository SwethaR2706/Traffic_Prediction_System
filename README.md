Traffic Prediction System (ML + Streamlit)

A machine learning web application that predicts highway traffic volume based on weather and time conditions.

Features

Predict traffic volume using machine learning

Convert predictions into traffic levels (Low / Moderate / Heavy)

Interactive Streamlit web interface

Uses real-world traffic and weather dataset

Tech Stack

Python

Streamlit

Scikit-learn

Pandas

NumPy

Matplotlib / Seaborn

How it Works

Traffic dataset is loaded and cleaned

Time features are extracted from timestamps

A Random Forest model is trained on traffic data

The trained model predicts traffic volume

Predictions are converted into traffic congestion levels

Streamlit interface allows users to input conditions and get predictions

Dataset

Metro Interstate Traffic Volume Dataset (UCI Machine Learning Repository)

Features used:

temp

rain_1h

snow_1h

clouds_all

hour

day

month

day_of_week

Target variable:

traffic_volume
Model Performance

MAE ≈ 132 vehicles
R² ≈ 0.38

Run the App

Install dependencies:

pip install -r requirements.txt

Start the application:

streamlit run app.py