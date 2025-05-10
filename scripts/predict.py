import pandas as pd
import mlflow
import os
from dotenv import load_dotenv

# Step 1: Load env variables
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("citibike_trip_prediction")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Step 2: Load feature data
df = pd.read_csv("./data/features/citibike_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

# Step 3: Select recent data for prediction
recent_df = df.groupby("start_station_id").tail(48)  # last 48 hours
feature_cols = [col for col in df.columns if col.startswith("lag_") or 
                col.startswith("rolling_") or col in ["hour", "weekday", "is_weekend"]]
X_recent = recent_df[feature_cols]

# Step 4: Load best model from MLflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("citibike_trip_prediction")
runs = client.search_runs(experiment.experiment_id, order_by=["metrics.mae ASC"], max_results=1)
best_run_id = runs[0].info.run_id
model_uri = f"runs:/{best_run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Step 5: Predict
recent_df["predicted_ride_count"] = model.predict(X_recent)

# Step 6: Save predictions
os.makedirs("./data/predictions", exist_ok=True)
recent_df[["start_station_id", "datetime", "predicted_ride_count"]].to_csv(
    "./data/predictions/predictions.csv", index=False
)
print("✅ Predictions saved to ./data/predictions/predictions.csv")
