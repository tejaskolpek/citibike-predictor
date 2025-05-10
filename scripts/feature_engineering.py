import os
import pandas as pd
from glob import glob

# Set input/output paths
input_dir = "./data/processed/"
output_dir = "./data/features/"
os.makedirs(output_dir, exist_ok=True)

# Find all processed CSVs
files = sorted(glob(os.path.join(input_dir, "*_processed.csv")))
if not files:
    raise FileNotFoundError(f"No processed CSV files found in {input_dir}")

# Read, convert, validate started_at column
df_list = []
for f in files:
    print(f"🔹 Reading file: {f}")
    df = pd.read_csv(f, nrows=10000)

    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.dropna(subset=["started_at"])
    print(f"✅ {len(df)} valid rows after datetime parsing")
    
    df_list.append(df)

# Combine all months
df = pd.concat(df_list).reset_index(drop=True)

# Create hourly datetime column
df["datetime"] = df["started_at"].dt.floor("H")

# Aggregate ride counts
agg_df = df.groupby(["start_station_id", "datetime"]).size().reset_index(name="ride_count")
agg_df.sort_values(["start_station_id", "datetime"], inplace=True)

# Lag features
for lag in range(1, 29):
    agg_df[f"lag_{lag}"] = agg_df.groupby("start_station_id")["ride_count"].shift(lag)

# Rolling statistics
grouped = agg_df.groupby("start_station_id")["ride_count"]
agg_df["rolling_mean_6"] = grouped.shift(1).rolling(6).mean().reset_index(0, drop=True)
agg_df["rolling_std_6"] = grouped.shift(1).rolling(6).std().reset_index(0, drop=True)
agg_df["rolling_mean_12"] = grouped.shift(1).rolling(12).mean().reset_index(0, drop=True)
agg_df["rolling_mean_24"] = grouped.shift(1).rolling(24).mean().reset_index(0, drop=True)

# Calendar features
agg_df["hour"] = agg_df["datetime"].dt.hour
agg_df["weekday"] = agg_df["datetime"].dt.weekday
agg_df["is_weekend"] = agg_df["weekday"].isin([5, 6]).astype(int)
agg_df["month"] = agg_df["datetime"].dt.month

# Final cleanup
final_df = agg_df.dropna().reset_index(drop=True)

# Save to CSV
output_file = os.path.join(output_dir, "citibike_features.csv")
final_df.to_csv(output_file, index=False)
print(f"✅ Feature file saved to {output_file}")
