import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# Check if the data is stationary using ADF test
def check_stationarity(series):
    result = adfuller(series)
    logging.info(f'ADF Statistic: {result[0]}, p-value: {result[1]}')
    print(f'ADF Statistic: {result[0]}, p-value: {result[1]}')
    if result[1] <= 0.05:
        logging.info('The time series is stationary.')
        print('The time series is stationary.')
    else:
        logging.info('The time series is not stationary. Consider differencing.')
        print('The time series is not stationary. Consider differencing.')

# Perform differencing to make the data stationary
def difference_data(series):
    return series.diff().dropna()

# Plot autocorrelation and partial autocorrelation
def plot_acf_pacf(series, lags=30):
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(series, lags=lags, ax=ax[0])
    plot_pacf(series, lags=lags, ax=ax[1])
    plt.show()

# Transform time series to supervised learning
def create_supervised_data(data, n_lag):
    X, y = [], []
    for i in range(n_lag, len(data)):
        X.append(data[i-n_lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Scale data
def scale_data(X, y):
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))
    
    return X_scaled, y_scaled, scaler_X, scaler_y

# Build and compile LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model
def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    logging.info('LSTM model training completed.')
    return history

def save_model(model, name, directory='models'):
    # Create the specified directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Generate a timestamp and file path
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = os.path.join(directory, f'lstm_model_{name}_{timestamp}.h5')
    
    # Save the model
    model.save(model_path)
    logging.info(f'Model saved as {model_path}')

def plot_loss(history):
    """Plot training loss history."""
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
