import streamlit as st
import pandas as pd
import plotly.express as px

# Load predictions
df = pd.read_csv("./data/predictions/predictions.csv", parse_dates=["datetime"])

st.set_page_config(page_title="Citi Bike Predictions", layout="wide")
st.title("🚲 Citi Bike Trip Predictions Dashboard")

# Sidebar - Station selection
station_ids = sorted(df["start_station_id"].unique())
selected_station = st.sidebar.selectbox("Select a Station", station_ids)

# Filter by selected station
station_df = df[df["start_station_id"] == selected_station]

# Line plot
fig = px.line(
    station_df,
    x="datetime",
    y="predicted_ride_count",
    title=f"Predicted Hourly Rides for Station {selected_station}",
    markers=True,
    labels={"predicted_ride_count": "Predicted Rides"}
)
st.plotly_chart(fig, use_container_width=True)

# Optional: Raw data table
with st.expander("📋 View Prediction Data"):
    st.dataframe(station_df.set_index("datetime"))
