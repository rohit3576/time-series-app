import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def preprocess_data(df):
    """Preprocess the uploaded data"""
    
    # Check if required columns exist
    if 'Date' not in df.columns:
        # Try to find date column
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                df = df.rename(columns={col: 'Date'})
                break
            except:
                continue
    
    # Check if value column exists
    if 'Value' not in df.columns and 'Sales' not in df.columns:
        # Try to find numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df = df.rename(columns={numeric_cols[0]: 'Value'})
    
    # Rename Sales to Value for consistency
    if 'Sales' in df.columns and 'Value' not in df.columns:
        df = df.rename(columns={'Sales': 'Value'})
    
    # Convert date and set index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')
    
    # Ensure we have a Value column
    if 'Value' not in df.columns:
        raise ValueError("No numeric column found for forecasting")
    
    # Handle missing values
    df = df.asfreq('D')
    df['Value'] = df['Value'].interpolate(method='time')
    
    # Remove any remaining NaN
    df = df.dropna()
    
    return df

def detect_seasonality(data, period=7):
    """Detect seasonality in time series"""
    if len(data) < 2 * period:
        return False
    
    # Use autocorrelation to detect seasonality
    autocorr = data.autocorr(lag=period)
    return abs(autocorr) > 0.3

def check_stationarity(data):
    """Check if time series is stationary using ADF test"""
    try:
        result = adfuller(data.dropna())
        return result[1] <= 0.05
    except:
        return False