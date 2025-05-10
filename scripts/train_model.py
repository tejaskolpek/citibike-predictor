import pandas as pd
import mlflow
import mlflow.sklearn
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import os
from dotenv import load_dotenv

# Step 1: Load env vars
load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("citibike_trip_prediction")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Step 2: Load data
df = pd.read_csv("./data/features/citibike_features.csv", parse_dates=["datetime"])
df = df.sort_values("datetime")

target_col = "ride_count"
feature_cols = [col for col in df.columns if col.startswith("lag_") or 
                col.startswith("rolling_") or col in ["hour", "weekday", "is_weekend"]]

split = int(len(df) * 0.8)
X_train, y_train = df[feature_cols][:split], df[target_col][:split]
X_test, y_test = df[feature_cols][split:], df[target_col][split:]

# Step 3: Train and log baseline model
with mlflow.start_run(run_name="baseline_model"):
    baseline_pred = [y_train.mean()] * len(y_test)
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    mlflow.log_metric("mae", baseline_mae)
    mlflow.log_param("model_type", "mean_baseline")

# Step 4: Train and log full LightGBM model
model1 = LGBMRegressor(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
mae1 = mean_absolute_error(y_test, pred1)

with mlflow.start_run(run_name="lightgbm_all_features"):
    mlflow.log_metric("mae", mae1)
    mlflow.log_params(model1.get_params())
    mlflow.sklearn.log_model(model1, "model")

# Step 5: Train and log top 10 feature model
importances = pd.Series(model1.feature_importances_, index=feature_cols).sort_values(ascending=False)
top10 = importances.head(10).index.tolist()
model2 = LGBMRegressor(n_estimators=100, random_state=42)
model2.fit(X_train[top10], y_train)
pred2 = model2.predict(X_test[top10])
mae2 = mean_absolute_error(y_test, pred2)

with mlflow.start_run(run_name="lightgbm_top10_features"):
    mlflow.log_metric("mae", mae2)
    mlflow.log_params(model2.get_params())
    mlflow.sklearn.log_model(model2, "model")

print("✅ Training and logging complete.")
