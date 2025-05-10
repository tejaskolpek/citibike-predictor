import pandas as pd
import mlflow
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Step 1: Load env variables
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("citibike_trip_prediction")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Step 2: Load feature data (for schema and station IDs)
df = pd.read_csv("./data/features/citibike_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Step 3: Create synthetic future datetime DataFrame
station_ids = df["start_station_id"].unique().tolist()
future_hours = [datetime(2025, 5, 10, 0) + timedelta(hours=i) for i in range(48)]
future_df = pd.DataFrame([
    {"start_station_id": sid, "datetime": dt}
    for sid in station_ids
    for dt in future_hours
])

# Step 4: Add calendar features
future_df["hour"] = future_df["datetime"].dt.hour
future_df["weekday"] = future_df["datetime"].dt.weekday
future_df["is_weekend"] = future_df["weekday"].isin([5, 6]).astype(int)
future_df["month"] = future_df["datetime"].dt.month

# Step 5: Add dummy lag/rolling features (zero-filled or averages)
avg_features = df.groupby("start_station_id").mean(numeric_only=True).reset_index()
avg_features = avg_features.drop(columns=["datetime", "ride_count"], errors='ignore')

future_df = future_df.merge(avg_features, on="start_station_id", how="left")

# Step 6: Load best model from MLflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("citibike_trip_prediction")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.mae ASC"], max_results=1)
best_run_id = runs[0].info.run_id
model_uri = f"runs:/{best_run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Step 7: Predict
feature_cols = [col for col in future_df.columns if col.startswith("lag_") or 
                col.startswith("rolling_") or col in ["hour", "weekday", "is_weekend", "month"]]
future_df["predicted_ride_count"] = model.predict(future_df[feature_cols])

# Step 8: Save predictions
os.makedirs("./data/predictions", exist_ok=True)
future_df[["start_station_id", "datetime", "predicted_ride_count"]].to_csv(
    "./data/predictions/predictions.csv", index=False
)
print("✅ Predictions saved to ./data/predictions/predictions.csv")
