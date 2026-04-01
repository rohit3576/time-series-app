import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocessing import preprocess_data

print("🚀 Starting EDA...\n")

# Load data
df = pd.read_csv("data/train.csv")

# Apply preprocessing
df = preprocess_data(df)

print("📊 DATA INFO:")
print(df.info())

print("\n📈 STATISTICS:")
print(df.describe())

# 📈 Time series plot
plt.figure(figsize=(10,5))
plt.plot(df['Value'])
plt.title("Time Series Plot")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid()
plt.show()

# 📊 Distribution
plt.figure(figsize=(6,4))
plt.hist(df['Value'], bins=30)
plt.title("Distribution of Values")
plt.grid()
plt.show()

# 📉 Rolling Mean (Trend)
df['Rolling Mean'] = df['Value'].rolling(window=7).mean()

plt.figure(figsize=(10,5))
plt.plot(df['Value'], label="Original")
plt.plot(df['Rolling Mean'], label="Rolling Mean (7 days)")
plt.legend()
plt.title("Trend Analysis")
plt.grid()
plt.show()

print("\n✅ EDA Completed Successfully!")