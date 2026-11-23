# Traffic Volume Prediction (ML + Streamlit)

Predicts hourly traffic volume on a US interstate (I-94, Minneapolis–St Paul) from weather and time of day, and maps the estimate to a Low / Moderate / Heavy traffic level through an interactive Streamlit app.

## Results (held-out future data)

Trained on 2012-10 → 2017-10, tested on the following year (2017-10 → 2018-09), so the model is never evaluated on hours near its training data.

| Model | MAE (vehicles/hr) | R² | Level accuracy* |
|---|---|---|---|
| Mean baseline | 1,724 | 0.00 | 0.29 |
| Linear Regression | 1,585 | 0.16 | 0.25 |
| **Random Forest** | **252** | **0.95** | **0.93** |

Forward-chaining 5-fold CV for the Random Forest: MAE ≈ 275, R² ≈ 0.94.
\*Share of test hours where the predicted Low/Moderate/Heavy level matches the actual level (<1,500 / <4,000 / ≥4,000).

**Key finding:** hour of day (~87% importance) and day of week (~11%) drive traffic; weather adds little in this dataset.

## Features
- Hourly volume prediction with a Random Forest regressor
- Traffic level classification (Low / Moderate / Heavy)
- Streamlit UI (temperature in °C, weather, hour, day of week, holiday toggle)

## Data
[Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (UCI): 48,204 raw rows, 40,565 after cleaning.

Cleaning: duplicate timestamps collapsed, 10 sensor-error rows (`temp = 0` K) removed, one 9,831 mm rain outlier clipped, `holiday` converted to a binary feature.
Features: `temp, rain_1h, snow_1h, clouds_all, hour, day, month, day_of_week, is_holiday`. Target: `traffic_volume`.

## Run
```
pip install -r requirements.txt
python train.py          # retrain, print metrics, save model/traffic_model.pkl
streamlit run app.py
```
Notebook with plots: `notebook/train_model.ipynb`.

## Limitations
- One highway sensor; the model does not generalise to other roads or cities.
- No lagged traffic features, so it estimates typical volume for a given hour rather than forecasting live conditions.
