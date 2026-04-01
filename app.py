import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import preprocess_data
from models.arima_model import run_arima
from models.rf_model import run_random_forest
from models.lstm_model import run_lstm

st.set_page_config(page_title="Forecast App", layout="wide")

st.title("📊 Time Series Forecasting (Future Prediction)")

# Sidebar
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ARIMA", "Random Forest", "LSTM"]
)

forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

# Upload
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    df = preprocess_data(df)

    st.subheader("📄 Processed Data")
    st.write(df.tail())

    # Plot original
    st.subheader("📈 Original Data")
    st.line_chart(df)

    if st.button("🔮 Predict Future"):

        train = df.copy()
        test = df.tail(1)  # dummy

        # Run selected model
        if model_choice == "ARIMA":
            forecast, _ = run_arima(train, test, forecast_days)

        elif model_choice == "Random Forest":
            forecast, _ = run_random_forest(train, test, forecast_days)

        else:
            forecast, _ = run_lstm(train, test, forecast_days)

        # Create future dates
        future_dates = pd.date_range(
            start=df.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days
        )

        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast
        }).set_index("Date")

        # 🔥 Plot combined
        st.subheader("📊 Forecast Graph")

        fig, ax = plt.subplots(figsize=(10,5))

        # Actual
        ax.plot(df.index, df['Value'], label="Actual Data")

        # Forecast
        ax.plot(forecast_df.index, forecast_df['Forecast'],
                label="Forecast", linestyle='dashed')

        ax.legend()
        ax.set_title(f"{model_choice} Forecast")

        st.pyplot(fig)

        # Show table
        st.subheader("🔢 Future Predictions")
        st.write(forecast_df)