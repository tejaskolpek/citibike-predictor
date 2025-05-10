import os
import pandas as pd
from glob import glob

# Load 10k rows per file
input_dir = "./data/processed/"
output_dir = "./data/features/"
os.makedirs(output_dir, exist_ok=True)

files = sorted(glob(os.path.join(input_dir, "*_processed.csv")))
df_list = [pd.read_csv(f, parse_dates=["started_at"], nrows=10000) for f in files]
df = pd.concat(df_list)
df["datetime"] = df["started_at"].dt.floor("H")

agg_df = df.groupby(["start_station_id", "datetime"]).size().reset_index(name="ride_count")
agg_df.sort_values(["start_station_id", "datetime"], inplace=True)

for lag in range(1, 29):
    agg_df[f"lag_{lag}"] = agg_df.groupby("start_station_id")["ride_count"].shift(lag)

grouped = agg_df.groupby("start_station_id")["ride_count"]
agg_df["rolling_mean_6"] = grouped.shift(1).rolling(6).mean().reset_index(0, drop=True)
agg_df["rolling_std_6"] = grouped.shift(1).rolling(6).std().reset_index(0, drop=True)
agg_df["rolling_mean_12"] = grouped.shift(1).rolling(12).mean().reset_index(0, drop=True)
agg_df["rolling_mean_24"] = grouped.shift(1).rolling(24).mean().reset_index(0, drop=True)

agg_df["hour"] = agg_df["datetime"].dt.hour
agg_df["weekday"] = agg_df["datetime"].dt.weekday
agg_df["is_weekend"] = agg_df["weekday"].isin([5, 6]).astype(int)
agg_df["month"] = agg_df["datetime"].dt.month

final_df = agg_df.dropna().reset_index(drop=True)
final_df.to_csv(os.path.join(output_dir, "citibike_features.csv"), index=False)
print("✅ Feature file saved.")
