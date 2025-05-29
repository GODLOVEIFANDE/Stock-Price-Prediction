import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

def predict_with_history(stock_name="CRDB", time_step=10):
    data_path = f"data/{stock_name}.csv"
    model_path = f"models/{stock_name}_lstm.h5"

    if not os.path.exists(data_path):
        return {"error": f"{stock_name} data not found"}

    # Load and sort
    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.sort_values("Date")
    df = df.dropna()

    # Ensure all required columns exist
    required_cols = ["Open", "High", "Low", "Close", "Date"]
    if not all(col in df.columns for col in required_cols):
        return {"error": f"Missing required columns in {stock_name}.csv"}

    # Scale Close price
    close_data = df[["Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:i + time_step, 0])
        y.append(scaled_data[i + time_step, 0])
    X = np.array(X).reshape(-1, time_step, 1)
    y = np.array(y)

    # Load or train model
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        os.makedirs("models", exist_ok=True)
        model.save(model_path)

    # Predict next closing price
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]

    # Get last 10 rows
    recent_df = df.tail(10).copy()
    recent_df = recent_df[["Date", "Open", "High", "Low", "Close"]]

    # Create predicted candle
    last_close = recent_df.iloc[-1]["Close"]
    predicted_date = recent_df.iloc[-1]["Date"] + pd.Timedelta(days=1)

    predicted_candle = {
        "Date": predicted_date,
        "Open": last_close,
        "High": max(last_close, pred_price * 1.01),
        "Low": min(last_close, pred_price * 0.99),
        "Close": pred_price
    }

    # Append prediction
    full_df = pd.concat([
        recent_df,
        pd.DataFrame([predicted_candle])
    ], ignore_index=True)

    # Convert to JSON-like dict
    return {
        "Date": full_df["Date"].dt.strftime('%Y-%m-%d').tolist(),
        "Open": full_df["Open"].round(2).tolist(),
        "High": full_df["High"].round(2).tolist(),
        "Low": full_df["Low"].round(2).tolist(),
        "Close": full_df["Close"].round(2).tolist()
    }
