import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def run_lstm(train, test, forecast_days, epochs=50):
    """Run LSTM model for time series forecasting"""
    
    try:
        scaler = MinMaxScaler()

        # 🔹 Combine for scaling
        all_data = pd.concat([train['Value'], test['Value']])
        scaler.fit(all_data.values.reshape(-1, 1))

        train_scaled = scaler.transform(train['Value'].values.reshape(-1, 1))

        # 🔹 Create sequences
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        seq_length = min(10, len(train_scaled) // 5)
        if seq_length < 2:
            seq_length = 2

        X_train, y_train = create_sequences(train_scaled, seq_length)

        if len(X_train) == 0:
            raise ValueError("Not enough data for LSTM")

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # 🔹 Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        early_stop = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )

        # 🔹 Train
        model.fit(
            X_train, y_train,
            epochs=min(epochs, 30),
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )

        # 🔹 Forecast
        last_sequence = train_scaled[-seq_length:].reshape(1, seq_length, 1)
        forecast_scaled = []

        for _ in range(forecast_days):
            pred = model.predict(last_sequence, verbose=0)
            forecast_scaled.append(pred[0, 0])

            last_sequence = np.append(
                last_sequence[:, 1:, :],
                pred.reshape(1, 1, 1),
                axis=1
            )

        # 🔹 Inverse scale
        forecast = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten()

        actual = test['Value'].values

        return forecast, actual

    except Exception as e:
        print(f"❌ LSTM error: {e}")

        # 🔥 Fallback
        last_avg = train['Value'].tail(7).mean()
        forecast = np.array([last_avg] * forecast_days)
        actual = test['Value'].values

        return forecast, actual