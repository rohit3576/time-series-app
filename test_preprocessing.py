import pandas as pd
from utils.preprocessing import preprocess_data, detect_seasonality, check_stationarity

print("🚀 Starting Preprocessing Test...\n")

# Load dataset
df = pd.read_csv("data/train.csv")

print("🔹 Raw Data:")
print(df.head())

# Apply preprocessing
processed_df = preprocess_data(df)

print("\n✅ Processed Data:")
print(processed_df.head())

# Insights
seasonal = detect_seasonality(processed_df['Value'])
stationary = check_stationarity(processed_df['Value'])

print("\n🧠 Insights:")
print("Seasonality:", "Yes 🔁" if seasonal else "No")
print("Stationary:", "Yes ✅" if stationary else "No ❌")

print("\n🎯 Done Successfully!")