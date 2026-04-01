import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def run_arima(train, test, forecast_days):
    """Run ARIMA model for time series forecasting"""
    
    try:
        train_data = train['Value']

        best_aic = float('inf')
        best_order = None

        # 🔥 Try combinations (optimized range)
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(train_data, order=(p, d, q))
                        result = model.fit()
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, d, q)
                    except:
                        continue

        # Default fallback
        if best_order is None:
            best_order = (1, 1, 0)

        print(f"✅ Best ARIMA Order: {best_order}")

        # Train final model
        model = ARIMA(train_data, order=best_order)
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=forecast_days)

        # Actual values
        actual = test['Value'].values

        return forecast.values, actual

    except Exception as e:
        print(f"❌ ARIMA error: {e}")

        # 🔥 Fallback (very important)
        last_avg = train['Value'].tail(7).mean()
        forecast = np.array([last_avg] * forecast_days)
        actual = test['Value'].values

        return forecast, actual