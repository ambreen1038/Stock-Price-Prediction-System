import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import os
import joblib now
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from train_lstm import train_lstm  # Import the train_lstm function
import matplotlib.pyplot as plt

# ------------------------------
# Sidebar: Navigation
# ------------------------------
st.sidebar.title("📊 Stock Price Prediction System")
page = st.sidebar.radio("Navigate", ["Home", "Dataset Visualization", "Trained Model Visualization", "Train Model", "Evaluation", "Predict New Values"])

# ------------------------------
# Common Functions
# ------------------------------
@st.cache_data
def load_stock_data(stock, start, end):
    """Fetch stock data using yfinance."""
    try:
        data = yf.download(stock, start=start, end=end)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def save_model_and_scaler(model, scaler, stock):
    """Save the trained model and scaler."""
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    st.success(f"Model and scaler saved as {model_path} and {scaler_path}.")

# ------------------------------
# Page 1: Home
# ------------------------------
if page == "Home":
    st.title("📈 Stock Price Prediction System")
    st.write("Welcome to the Stock Price Prediction System! Use the sidebar to navigate through the app.")

    # User Inputs
    stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-21'))

    # Load Data
    data = load_stock_data(stock, start_date, end_date)
    if data is not None:
        st.write(f"### 📊 Stock Data for {stock}")
        st.dataframe(data)

        # Download CSV
        st.download_button(
            label="📥 Download Stock Data",
            data=data.to_csv(index=False),
            file_name=f"{stock}_stock_data.csv",
            mime="text/csv"
        )

# ------------------------------
# Page 2: Dataset Visualization
# ------------------------------
elif page == "Dataset Visualization":
    st.title("📊 Dataset Visualization")

    # User Inputs
    stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-21'))

    # Load Data
    data = load_stock_data(stock, start_date, end_date)
    if data is not None:
        st.write(f"### 📈 Stock Data for {stock}")

        # Calculate Moving Averages
        data['MA_50'] = data['Close'].rolling(50).mean()
        data['MA_100'] = data['Close'].rolling(100).mean()
        data['MA_200'] = data['Close'].rolling(200).mean()

        # Matplotlib Price vs MA50
        st.write("#### Price vs MA50")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        ax.plot(data['Date'], data['MA_50'], label='MA50', color='orange')
        ax.set_title(f"{stock} Price vs MA50")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Matplotlib Price vs MA50 vs MA100
        st.write("#### Price vs MA50 vs MA100")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        ax.plot(data['Date'], data['MA_50'], label='MA50', color='orange')
        ax.plot(data['Date'], data['MA_100'], label='MA100', color='green')
        ax.set_title(f"{stock} Price vs MA50 vs MA100")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Matplotlib Price vs MA100 vs MA200
        st.write("#### Price vs MA100 vs MA200")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data['Date'], data['Close'], label='Close Price', color='blue')
        ax.plot(data['Date'], data['MA_100'], label='MA100', color='green')
        ax.plot(data['Date'], data['MA_200'], label='MA200', color='red')
        ax.set_title(f"{stock} Price vs MA100 vs MA200")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

# ------------------------------
# Page 3: Train Model
# ------------------------------
elif page == "Train Model":
    st.title("🚀 Train a New Model")

    # User Inputs
    stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-21'))

    # Train Model
    if st.button("Train Model"):
        with st.spinner("Training model... This may take a few minutes."):

            try:
                model, scaler = train_lstm(stock, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), f"{stock}_lstm_model.keras")
                save_model_and_scaler(model, scaler, stock)
                st.success("Model training complete!")

                # Display Model Summary
                st.write("### Model Summary")
                model.summary(print_fn=lambda x: st.text(x))

                # Download Model
                st.download_button(
                    label="📥 Download Trained Model",
                    data=open(f"{stock}_lstm_model.keras", "rb").read(),
                    file_name=f"{stock}_lstm_model.keras",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Error during training: {e}")
# ------------------------------
# Page 4: Evaluation
# ------------------------------
elif page == "Evaluation":
    st.title("🔍 Model Evaluation")

    # User Inputs
    stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"])
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Loaded pretrained model and scaler.")

        # Load Data
        start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
        end_date = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-21'))
        data = load_stock_data(stock, start_date, end_date)

        if data is not None:
            # Feature Engineering
            data['MA_100'] = data['Close'].rolling(100).mean()
            data['MA_200'] = data['Close'].rolling(200).mean()
            data.dropna(inplace=True)

            features = ['Close', 'MA_100', 'MA_200']
            test_scaled = scaler.transform(data[features])

            # Create Sequences
            def create_sequences(data, time_steps=100):
                x, y = [], []
                for i in range(time_steps, len(data)):
                    x.append(data[i - time_steps:i])
                    y.append(data[i, 0])  # Predicting 'Close' price
                return np.array(x), np.array(y)

            x_test, y_test = create_sequences(test_scaled)

            # Model Prediction
            y_pred_scaled = model.predict(x_test)
            y_pred = y_pred_scaled * (1 / scaler.scale_[0])  # Inverse transform

            # Evaluation Metrics
            st.write("### Evaluation Metrics")
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
            st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
            st.write(f"**R² Score:** {r2:.4f}")

            # Model Summary
            st.write("### Model Summary")
            model.summary(print_fn=lambda x: st.text(x))
    else:
        st.warning("No pretrained model found. Please train a model first.")
# ------------------------------
# Page 5: Predict New Values
# ------------------------------
elif page == "Predict New Values":
    st.title("🔮 Predict New Values")

    # User Inputs
    stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"])
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Loaded pretrained model and scaler.")

        # User Input for Prediction
        st.write("### Enter Input Values for Prediction")
        close = st.number_input("Close Price", value=100.0)
        ma_100 = st.number_input("100-Day Moving Average", value=100.0)
        ma_200 = st.number_input("200-Day Moving Average", value=100.0)

        if st.button("Predict"):
            input_data = np.array([[close, ma_100, ma_200]])
            input_scaled = scaler.transform(input_data)
            input_scaled = input_scaled.reshape((1, 1, 3))  # Reshape for LSTM input
            prediction_scaled = model.predict(input_scaled)
            prediction = prediction_scaled * (1 / scaler.scale_[0])  # Inverse transform
            st.success(f"Predicted Close Price: {prediction[0][0]:.2f}")

            # Visualization
            st.write("### Prediction Visualization")
            fig = px.line(title=f"Predicted Close Price for {stock}")
            fig.add_scatter(x=[0], y=[prediction[0][0]], mode='markers', name='Predicted Price', marker=dict(color="red", size=10))
            st.plotly_chart(fig)

            # Load stock data for original vs predicted price graph
            start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
            end_date = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-21'))
            data = load_stock_data(stock, start_date, end_date)

            if data is not None:
                # Feature Engineering for the model input
                data['MA_100'] = data['Close'].rolling(100).mean()
                data['MA_200'] = data['Close'].rolling(200).mean()
                data.dropna(inplace=True)

                # Scale the data for testing
                data_test_scaled = scaler.transform(data[['Close', 'MA_100', 'MA_200']].values)

                # Create sequences for prediction
                x = []
                y = []
                for i in range(100, data_test_scaled.shape[0]):
                    x.append(data_test_scaled[i-100:i])
                    y.append(data_test_scaled[i, 0])  # Close price

                x, y = np.array(x), np.array(y)

                # Predict using the model
                predict = model.predict(x)

                # Reverse scaling
                scale = 1 / scaler.scale_[0]
                predict = predict * scale
                y = y * scale

                # Plot Original vs Predicted Prices
                st.subheader('Original Price vs Predicted Price')
                fig4 = plt.figure(figsize=(8, 6))
                plt.plot(predict, 'r', label='Predicted Price')
                plt.plot(y, 'g', label='Original Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig4)

    else:
        st.warning("No pretrained model found. Please train a model first.")

elif page == "Trained Model Visualization":
    st.title("📊 Trained Model Visualization")

    st.write("""
    This section shows the **Original Price vs Predicted Price** graph for the data from the test dataset. The graph compares the actual stock prices with the predictions made by the trained LSTM model.
    """)

    # User Inputs
    stock = st.sidebar.selectbox("Select Stock", ["GOOG", "AAPL", "AMZN", "MSFT"])
    model_path = f"{stock}_lstm_model.keras"
    scaler_path = f"{stock}_scaler.save"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        st.success("Loaded pretrained model and scaler.")

        # Load Test Data
        start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2012-01-01'))
        end_date = st.sidebar.date_input("End Date", pd.to_datetime('2022-12-21'))
        data = load_stock_data(stock, start_date, end_date)

        if data is not None:
            # Feature Engineering
            data['MA_100'] = data['Close'].rolling(100).mean()
            data['MA_200'] = data['Close'].rolling(200).mean()
            data.dropna(inplace=True)

            features = ['Close', 'MA_100', 'MA_200']
            data_test_scaled = scaler.transform(data[features].values)

            # Create Sequences
            x_test, y_test = [], []
            for i in range(100, data_test_scaled.shape[0]):
                x_test.append(data_test_scaled[i-100:i])
                y_test.append(data_test_scaled[i, 0])
            
            x_test, y_test = np.array(x_test), np.array(y_test)

            # Model Prediction
            y_pred_scaled = model.predict(x_test)
            y_pred = y_pred_scaled * (1 / scaler.scale_[0])  # Inverse transform

            # Plot the graph
            st.subheader('Original Price vs Predicted Price')
            fig4 = plt.figure(figsize=(8, 6))
            plt.plot(y_pred, 'r', label='Predicted Price')
            plt.plot(y_test, 'g', label='Original Price')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig4)
    else:
        st.warning("No pretrained model found. Please train a model first.")