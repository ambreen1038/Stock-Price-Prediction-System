# 📈 Stock Price Prediction using LSTM

This project predicts stock prices using a deep learning model built with LSTM (Long Short-Term Memory) networks. It leverages historical stock data fetched via Yahoo Finance and applies multiple moving averages to enhance predictions. The model is trained on scaled and sequenced data and then saved along with the scaler for future inference.

## 🔧 Features
- Fetches historical stock data using `yfinance`
- Calculates 100-day and 200-day moving averages
- Scales data using `MinMaxScaler`
- Builds and trains a deep LSTM model
- Saves both the trained model and the scaler for future use

## 🛠 Tech Stack
- Python
- NumPy, Pandas
- yFinance
- Scikit-learn
- Keras (TensorFlow backend)
- Joblib