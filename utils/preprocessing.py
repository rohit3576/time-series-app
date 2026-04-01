import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def preprocess_data(df):
    """Preprocess the uploaded data"""

    # ✅ STEP 1: Detect Date column safely
    possible_date_cols = ['Date', 'Order Date', 'order_date', 'date']

    for col in possible_date_cols:
        if col in df.columns:
            df = df.rename(columns={col: 'Date'})
            break

    # Fallback: detect column containing 'date'
    if 'Date' not in df.columns:
        for col in df.columns:
            if 'date' in col.lower():
                df = df.rename(columns={col: 'Date'})
                break

    if 'Date' not in df.columns:
        raise ValueError("No date column found")

    # ✅ STEP 2: Detect Value column
    if 'Value' not in df.columns and 'Sales' not in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df = df.rename(columns={numeric_cols[0]: 'Value'})

    # Rename Sales → Value
    if 'Sales' in df.columns and 'Value' not in df.columns:
        df = df.rename(columns={'Sales': 'Value'})

    if 'Value' not in df.columns:
        raise ValueError("No numeric column found for forecasting")

    # ✅ STEP 3: Convert Date properly
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    # ✅ STEP 4: Sort + Set index
    df = df.sort_values('Date')
    df = df.set_index('Date')

    # 🔥 IMPORTANT: Aggregate data (for sales dataset)
    df = df.groupby('Date')['Value'].sum().to_frame()

    # ✅ STEP 5: Handle missing dates
    df = df.asfreq('D')
    df['Value'] = df['Value'].interpolate(method='time')

    # Remove any remaining NaN
    df = df.dropna()

    return df


def detect_seasonality(data, period=7):
    """Detect seasonality in time series"""
    if len(data) < 2 * period:
        return False

    autocorr = data.autocorr(lag=period)
    return abs(autocorr) > 0.3


def check_stationarity(data):
    """Check if time series is stationary using ADF test"""
    try:
        result = adfuller(data.dropna())
        return result[1] <= 0.05
    except:
        return False