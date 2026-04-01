import pandas as pd
from utils.preprocessing import preprocess_data
from models.lstm_model import run_lstm

print("🚀 Testing LSTM Model...\n")

# Load data
df = pd.read_csv("data/train.csv")

# Preprocess
df = preprocess_data(df)

# Split
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

forecast_days = len(test)

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

# Run LSTM
forecast, actual = run_lstm(train, test, forecast_days)

print("\n✅ Results:")
print("LSTM Forecast (first 5):", forecast[:5])
print("Actual (first 5):", actual[:5])