import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import mlflow

# Load environment variables
load_dotenv()
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Load the trained LightGBM model (top 10 features for this example)
model_uri = "models:/lightgbm_top10_features/Production"
model = mlflow.pyfunc.load_model(model_uri)

# Load historical feature data to extract station IDs and lags
df = pd.read_csv("./data/features/citibike_features.csv", parse_dates=["datetime"])
df["start_station_id"] = df["start_station_id"].astype(str)

# Define future datetime range for May 10-11, 2025
station_ids = df["start_station_id"].unique().tolist()
future_hours = [datetime(2025, 5, 10) + timedelta(hours=i) for i in range(48)]
future_df = pd.DataFrame([
    {"start_station_id": sid, "datetime": dt}
    for sid in station_ids
    for dt in future_hours
])

# Merge lag stats by grabbing recent values (fill with mean if missing)
last_stats = df.groupby("start_station_id").tail(1).set_index("start_station_id")

# Fill static or temporal features
future_df["hour"] = future_df["datetime"].dt.hour
future_df["weekday"] = future_df["datetime"].dt.weekday
future_df["is_weekend"] = future_df["weekday"].isin([5, 6]).astype(int)
future_df["month"] = future_df["datetime"].dt.month

# Fill lag/rolling features with fallback values
feature_cols = [c for c in df.columns if c not in ["datetime", "ride_count"]]

for col in feature_cols:
    if col in future_df.columns:
        continue
    future_df[col] = future_df["start_station_id"].map(last_stats[col] if col in last_stats.columns else 0)

# Reorder columns
predict_cols = [c for c in future_df.columns if c not in ["datetime", "start_station_id"]]
X_future = future_df[predict_cols]

# Predict
future_df["predicted_ride_count"] = model.predict(X_future)

# Save
os.makedirs("./data/predictions", exist_ok=True)
future_df.to_csv("./data/predictions/predictions.csv", index=False)
print("✅ 2025 Forecast Saved to predictions.csv")
