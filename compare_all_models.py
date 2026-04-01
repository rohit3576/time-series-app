import pandas as pd
import numpy as np

from utils.preprocessing import preprocess_data
from models.arima_model import run_arima
from models.rf_model import run_random_forest
from models.lstm_model import run_lstm

print("🚀 Comparing ALL Models...\n")

# Load data
df = pd.read_csv("data/train.csv")
df = preprocess_data(df)

# Split
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

forecast_days = len(test)

# Run models
print("Running ARIMA...")
arima_forecast, actual = run_arima(train, test, forecast_days)

print("Running RF...")
rf_forecast, _ = run_random_forest(train, test, forecast_days)

print("Running LSTM...")
lstm_forecast, _ = run_lstm(train, test, forecast_days)

# 📊 Metrics
def evaluate(actual, predicted):
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mae, rmse

arima_mae, arima_rmse = evaluate(actual, arima_forecast)
rf_mae, rf_rmse = evaluate(actual, rf_forecast)
lstm_mae, lstm_rmse = evaluate(actual, lstm_forecast)

print("\n📊 RESULTS:")
print(f"ARIMA → MAE: {arima_mae:.2f}, RMSE: {arima_rmse:.2f}")
print(f"RF    → MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
print(f"LSTM  → MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}")

# 🏆 Best model
results = {
    "ARIMA": arima_rmse,
    "Random Forest": rf_rmse,
    "LSTM": lstm_rmse
}

best_model = min(results, key=results.get)

print(f"\n🏆 Best Model: {best_model}")