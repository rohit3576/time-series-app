import pandas as pd
from utils.preprocessing import preprocess_data
from models.rf_model import run_random_forest

print("🚀 Testing Random Forest Model...\n")

# Load data
df = pd.read_csv("data/train.csv")

# Preprocess
df = preprocess_data(df)

# Split data
split = int(len(df) * 0.8)
train = df[:split]
test = df[split:]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

forecast_days = len(test)

# Run RF
forecast, actual = run_random_forest(train, test, forecast_days)

print("\n✅ Results:")
print("RF Forecast (first 5):", forecast[:5])
print("Actual (first 5):", actual[:5])