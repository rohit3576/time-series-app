import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def run_random_forest(train, test, forecast_days):
    """Run Random Forest model for time series forecasting"""
    
    try:
        # Create lag features
        def create_features(data, lags=7):
            df = data.copy()
            
            # Add lag features
            for i in range(1, min(lags + 1, len(df))):
                df[f'lag_{i}'] = df['Value'].shift(i)
            
            # Add rolling statistics
            df['rolling_mean_3'] = df['Value'].rolling(window=3).mean()
            df['rolling_mean_7'] = df['Value'].rolling(window=7).mean()
            df['rolling_std_3'] = df['Value'].rolling(window=3).std()
            
            # Add day of week and month features
            if hasattr(df.index, 'dayofweek'):
                df['day_of_week'] = df.index.dayofweek
                df['month'] = df.index.month
                df['day_of_year'] = df.index.dayofyear
            
            # Drop NaN values
            df = df.dropna()
            
            return df
        
        # Prepare training data
        train_feat = create_features(train)
        
        if len(train_feat) < 10:
            # Not enough data, use simple approach
            last_value = train['Value'].iloc[-1]
            return np.array([last_value] * forecast_days), test['Value'].values
        
        # Separate features and target
        feature_cols = [col for col in train_feat.columns if col != 'Value']
        X_train = train_feat[feature_cols]
        y_train = train_feat['Value']
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # Generate forecast recursively
        last_data = train.copy()
        forecast = []
        
        for i in range(forecast_days):
            # Create features for prediction
            features = create_features(last_data)
            
            if len(features) > 0:
                # Get latest features
                latest_features = features.iloc[[-1]]
                X_pred = latest_features[feature_cols]
                X_pred_scaled = scaler.transform(X_pred)
                
                # Make prediction
                pred = rf_model.predict(X_pred_scaled)[0]
                forecast.append(pred)
                
                # Add to data for next prediction
                new_index = last_data.index[-1] + pd.Timedelta(days=1)
                new_row = pd.DataFrame({'Value': [pred]}, index=[new_index])
                last_data = pd.concat([last_data, new_row])
            else:
                forecast.append(last_data['Value'].iloc[-1])
        
        # Get actual values for test period
        actual = test['Value'].values
        
        return np.array(forecast), actual
    
    except Exception as e:
        print(f"Random Forest error: {e}")
        # Fallback: simple average
        last_values = train['Value'].tail(7).mean()
        forecast = np.array([last_values] * forecast_days)
        actual = test['Value'].values
        return forecast, actual