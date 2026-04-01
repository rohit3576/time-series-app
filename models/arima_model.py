import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

def run_arima(train, test, forecast_days):
    """Run ARIMA model for time series forecasting"""
    
    try:
        # Get the time series data
        train_data = train['Value']
        
        # Auto-select ARIMA order (simplified for speed)
        best_aic = float('inf')
        best_order = None
        
        # Try different combinations
        for p in range(0, 4):
            for d in range(0, 2):
                for q in range(0, 4):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        result = model.fit()
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        # If no order found, use default
        if best_order is None:
            best_order = (1, 1, 0)
        
        # Fit the best model
        model = ARIMA(train_data, order=best_order)
        model_fit = model.fit()
        
        # Generate forecast
        forecast = model_fit.forecast(steps=forecast_days)
        
        # Get actual values for test period
        actual = test['Value'].values
        
        return forecast.values, actual
    
    except Exception as e:
        print(f"ARIMA error: {e}")
        # Fallback: simple moving average
        last_values = train['Value'].tail(7).mean()
        forecast = np.array([last_values] * forecast_days)
        actual = test['Value'].values
        return forecast, actual