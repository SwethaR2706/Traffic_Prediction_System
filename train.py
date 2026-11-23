"""Train and evaluate the traffic volume model (reproducible script)."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ["temp", "rain_1h", "snow_1h", "clouds_all",
            "hour", "day", "month", "day_of_week", "is_holiday"]
LOW, HIGH = 1500, 4000          # Low < 1500 <= Moderate < 4000 <= Heavy


def load():
    df = pd.read_csv("dataset/Metro_Interstate_Traffic_Volume.csv.gz")
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.drop_duplicates(subset="date_time", keep="first")   # duplicate timestamps
    df = df[df["temp"] > 0]                                       # 0 K = sensor error
    df["rain_1h"] = df["rain_1h"].clip(upper=df["rain_1h"].quantile(0.9999))  # one 9831 mm outlier
    df["is_holiday"] = df["holiday"].notna().astype(int)          # keep holiday as a feature
    df["hour"] = df.date_time.dt.hour
    df["day"] = df.date_time.dt.day
    df["month"] = df.date_time.dt.month
    df["day_of_week"] = df.date_time.dt.dayofweek
    return df.sort_values("date_time").reset_index(drop=True)


def level(v):
    return np.where(v < LOW, 0, np.where(v < HIGH, 1, 2))


def evaluate(model, tr, te):
    model.fit(tr[FEATURES], tr.traffic_volume)
    p = model.predict(te[FEATURES])
    y = te.traffic_volume.values
    return dict(MAE=mean_absolute_error(y, p), R2=r2_score(y, p),
                level_acc=accuracy_score(level(y), level(p)))


if __name__ == "__main__":
    df = load()
    print("rows after cleaning:", len(df))
    n = int(len(df) * 0.8)                       # chronological 80/20 split
    tr, te = df.iloc[:n], df.iloc[n:]
    print("train:", tr.date_time.min().date(), "->", tr.date_time.max().date())
    print("test :", te.date_time.min().date(), "->", te.date_time.max().date())

    models = {
        "Mean baseline": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=10,
                                               random_state=42, n_jobs=-1),
    }
    results = {k: evaluate(m, tr, te) for k, m in models.items()}
    for k, r in results.items():
        print(f"{k:18s} MAE {r['MAE']:7.1f}  R2 {r['R2']:.3f}  level-acc {r['level_acc']:.3f}")

    # 5-fold forward-chaining CV for the chosen model
    cv = []
    for a, b in TimeSeriesSplit(n_splits=5).split(df):
        m = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
        cv.append(evaluate(m, df.iloc[a], df.iloc[b]))
    cv_mae = np.mean([c["MAE"] for c in cv]); cv_r2 = np.mean([c["R2"] for c in cv])
    print(f"Forward-chaining 5-fold CV: MAE {cv_mae:.1f}  R2 {cv_r2:.3f}")

    rf = models["Random Forest"]
    imp = dict(sorted(zip(FEATURES, rf.feature_importances_.round(3)), key=lambda x: -x[1]))
    print("feature importance:", imp)

    # final model: refit on all data for the app
    final = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
    final.fit(df[FEATURES], df.traffic_volume)
    joblib.dump(final, "model/traffic_model.pkl", compress=3)
    json.dump({"holdout": results, "cv_mae": cv_mae, "cv_r2": cv_r2, "importance": imp,
               "features": FEATURES}, open("model/metrics.json", "w"), indent=2, default=float)
    print("saved model/traffic_model.pkl")
