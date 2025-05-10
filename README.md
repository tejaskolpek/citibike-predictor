# 🚲 Citi Bike Trip Prediction System

This project predicts hourly Citi Bike ride demand in NYC using historical trip data. It is an end-to-end machine learning pipeline built with Python, GitHub Actions, MLflow (via DagsHub), and deployed using Streamlit.

---

## 📊 Project Overview

The goal is to forecast the number of rides starting from each Citi Bike station on an hourly basis using time-series and calendar features.

* **Data Source**: [Citi Bike System Data](https://citibikenyc.com/system-data)
* **Prediction Target**: Ride count per station per hour
* **Tech Stack**: Python, Pandas, LightGBM, Hopsworks, DagsHub MLflow, Streamlit, GitHub Actions

---

## 🗂️ Project Structure

```
citibike-predictor/
├── .github/workflows/         # GitHub Actions for automation
│   ├── feature_engineering.yml
│   ├── train_model.yml
│   └── predict.yml
├── data/
│   ├── processed/             # Sample processed CSV (10K rows)
│   ├── features/              # Generated features (output of pipeline)
│   └── predictions/           # Final predicted ride counts
├── scripts/
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── predict.py
├── streamlit_app/
│   └── app.py                 # Dashboard for visualizing predictions
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ ML Pipeline Steps

1. **Data Preprocessing**

   * Raw monthly trip data cleaned
   * Limited to 10,000 rows per month
   * Top 3 stations selected

2. **Feature Engineering**

   * Lag features (1–28 hours)
   * Rolling mean/std (6, 12, 24 hours)
   * Calendar features (hour, weekday, weekend)

3. **Model Training**

   * Baseline (hourly mean)
   * LightGBM with all features
   * LightGBM with top 10 features

4. **Experiment Tracking**

   * Models logged to MLflow via DagsHub
   * MAE tracked across all runs

5. **Batch Prediction**

   * Best model loads features
   * Outputs predictions for next 48 hours
   * Results saved to `/data/predictions/`

6. **Streamlit App**

   * Displays predicted ride counts per station
   * Interactive visualization

---

## 🚀 GitHub Actions

| Workflow                  | Trigger       | Purpose                          |
| ------------------------- | ------------- | -------------------------------- |
| `feature_engineering.yml` | Manual / Push | Generate lag + calendar features |
| `train_model.yml`         | Manual / Push | Train 3 models and log to MLflow |
| `predict.yml`             | Manual / Push | Generate and store predictions   |

---

## 🥪 How to Run Locally

1. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/<your-username>/citibike-predictor.git
   cd citibike-predictor
   pip install -r requirements.txt
   ```

2. Set up your `.env` file using `.env.example`

3. Run the pipeline manually:

   ```bash
   python scripts/feature_engineering.py
   python scripts/train_model.py
   python scripts/predict.py
   ```

4. Launch the Streamlit app:

   ```bash
   streamlit run streamlit_app/app.py
   ```

---

## 🌐 Demo

[Streamlit Dashboard](https://citibike-lnsvtvkcl6kxmcngbmfezp.streamlit.app/)
[GitHub Repo](https://github.com/tejasklolpek/citibike-predictor)

---

## 👨‍🏫 TA Review Tips

* All 3 GitHub Actions are active and tested
* Model performance improves over baseline (check MLflow on DagsHub)
* Streamlit app live and interpretable
* Slides include pipeline, metrics, and GitHub proof

---

## 🏁 Credits

* Developed for AML500 Final Project
* Guided by NYC Taxi Ride Forecasting Pipeline
* Powered by open data & open-source ML tools
