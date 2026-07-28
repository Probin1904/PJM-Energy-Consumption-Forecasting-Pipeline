# ⚡ PJM Energy Consumption Forecasting

An end-to-end machine learning pipeline that forecasts **hourly electricity consumption** using the PJM Hourly Energy Consumption dataset (AEP region). The project covers the full workflow — data cleaning, exploratory data analysis, feature engineering, model training/evaluation, and feature importance analysis — with the trained model prepared for deployment.

---

## 📌 Project Overview

Electricity providers need accurate short-term load forecasts to balance supply and demand efficiently. This project builds and compares multiple regression models to predict hourly energy consumption (in MW) for American Electric Power (AEP), using historical load data spanning **2004–2018**.

## 🎯 Objectives

- Clean and preprocess historical hourly energy consumption data
- Explore consumption trends, daily/weekly/seasonal patterns
- Engineer time-based and lag/rolling features for forecasting
- Train and compare multiple ML models: Linear Regression, Random Forest, XGBoost
- Evaluate models using MAE, RMSE, and R² Score
- Analyze feature importance to understand what drives predictions
- Save the best-performing model for deployment

## 📊 Dataset

- **Source:** [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (Kaggle)
- **File used:** `AEP_hourly.csv`
- **Rows:** ~121,000 hourly readings
- **Range:** October 2004 – August 2018
- **Target variable:** `AEP_MW` — hourly energy consumption in megawatts

## 🛠️ Tech Stack

`Python` • `Pandas` • `NumPy` • `Matplotlib` • `Seaborn` • `Scikit-learn` • `XGBoost` • `Joblib`

## 🔄 Pipeline / Notebook Structure

| Phase | Description |
|---|---|
| **1. Data Understanding** | Initial inspection of shape, dtypes, missing values, and summary statistics |
| **2. Data Cleaning & Preprocessing** | Datetime parsing, sorting, duplicate removal, missing value handling |
| **3. Exploratory Data Analysis** | Time series trends, weekday/weekend histograms, monthly bar chart, daily load profile, day-of-week box plot, seasonal scatter plot, hour × month heatmap, 30-day rolling average trend |
| **4. Feature Engineering** | Calendar features (Hour, Day, Month, Year, DayOfWeek, Quarter, IsWeekend) + lag/rolling features (`lag_1`, `lag_24`, `lag_168`, `roll_mean_24`) |
| **5. Model Training** | Linear Regression, Random Forest Regressor, XGBoost Regressor |
| **6. Model Evaluation** | MAE, RMSE, and R² comparison across models |
| **7. Feature Importance Analysis** | Per-model importance charts, grouped (lag vs. calendar) importance, and an ablation study without `lag_1` |

## 📈 Results

Chronological 80/20 train-test split (train: 2004–2015, test: 2015–2018).

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| Linear Regression | 379.77 | 482.96 | 0.9610 |
| Random Forest | 144.37 | 196.60 | 0.9935 |
| **XGBoost (Best Model)** | **126.13** | **170.70** | **0.9951** |

**Key finding:** Lag/rolling features (recent actual consumption history) account for **~97%** of XGBoost's predictive power, versus ~3% for calendar features alone — expected, since hourly electricity demand is highly autocorrelated. As a robustness check, removing `lag_1` (previous hour's value) drops R² to **0.9228**, still strong, driven by `lag_24`, `lag_168`, and `roll_mean_24`.

> ⚠️ **Note on R²:** This high score reflects a 1-hour-ahead forecast, which benefits heavily from short-term autocorrelation. A day-ahead or multi-step forecast (dropping `lag_1`) is a harder, more realistic deployment scenario and would be expected to score lower — see Future Work.

## 📂 Repository Structure

```
├── PJM_Energy_Consumption_Forecasting_using_ML.ipynb   # Main notebook (full pipeline)
├── AEP_hourly.csv                                       # Dataset (hourly AEP consumption)
└── README.md                                            # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### Run
1. Clone/download this repository
2. Place `AEP_hourly.csv` in the project root (or update the path in the notebook's data-loading cell)
3. Open and run `PJM_Energy_Consumption_Forecasting_using_ML.ipynb` top to bottom

## 🔮 Future Work

- **Day-ahead forecasting:** Drop `lag_1`, add `roll_mean_168`, and restructure the target for a realistic multi-step-ahead forecast
- **Weather integration:** Add temperature/weather forecast features — likely the strongest remaining lever for accuracy beyond lag features
- **Deployment:** Serve the saved model via a REST API (FastAPI/Flask) with a simple frontend dashboard for visualizing predicted vs. actual load

## 📄 License

This project uses the publicly available PJM Hourly Energy Consumption dataset. Check the original [Kaggle dataset page](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) for licensing details.


