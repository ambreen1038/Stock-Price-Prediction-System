import numpy as np
import pandas as pd
import yfinance as yf
import os
import joblib  # For saving the scaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM

# Function to fetch stock data
def fetch_stock_data(stock, start, end):
    data = yf.download(stock, start=start, end=end)
    data.reset_index(inplace=True)
    return data

# Function to preprocess data for LSTM
def preprocess_data(data, time_steps=100):
    # Add moving averages
    data['MA_100'] = data['Close'].rolling(100).mean()
    data['MA_200'] = data['Close'].rolling(200).mean()
    data.dropna(inplace=True)

    # Train-test split
    train_size = int(len(data) * 0.80)
    data_train = data.iloc[:train_size]
    data_test = data.iloc[train_size:]

    # Feature selection
    features = ['Close', 'MA_100', 'MA_200']
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(data_train[features])
    test_scaled = scaler.transform(data_test[features])

    # Creating sequences
    def create_sequences(data):
        x, y = [], []
        for i in range(time_steps, len(data)):
            x.append(data[i - time_steps:i])
            y.append(data[i, 0])  # Predicting 'Close' price
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(train_scaled)
    x_test, y_test = create_sequences(test_scaled)

    return x_train, y_train, x_test, y_test, scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=60, activation='relu', return_sequences=True),
        Dropout(0.3),
        LSTM(units=80, activation='relu', return_sequences=True),
        Dropout(0.4),
        LSTM(units=120, activation='relu'),
        Dropout(0.5),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the model
def train_lstm(stock, start, end, save_path):
    print(f"Fetching stock data for {stock}...")
    data = fetch_stock_data(stock, start, end)

    print("Preprocessing data...")
    x_train, y_train, x_test, y_test, scaler = preprocess_data(data)

    print("Building model...")
    model = build_lstm_model((x_train.shape[1], x_train.shape[2]))

    print("Training model...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

    print(f"Saving model to {save_path}...")
    model.save(save_path)

    # Save the scaler
    scaler_path = f"{stock}_scaler.save"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    return model, scaler

# Run the script independently
# if __name__ == "__main__":
    # stock = "GOOG"
    # start = "2012-01-01"
    # end = "2022-12-21"
    # save_path = f"{stock}_lstm_model.keras"
    
    # train_lstm(stock, start, end, save_path)
    # print("Training complete!")