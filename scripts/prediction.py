from sklearn.preprocessing import StandardScaler
import pandas as pd
import logging

def preprocess_data(data):
    """Preprocesses the data for modeling."""
    logger = logging.getLogger(__name__)
    # Convert date columns
    logger.info("Converting date columns to datetime format.")
    data['Date'] = pd.to_datetime(data['Date'])

    # Extract features
    logger.info("Extracting additional features.")
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6])  # Assuming 5 and 6 represent weekends
    # Adjust DaysToNearestHoliday and DaysAfterHoliday based on actual holiday data
    data['DaysToNearestHoliday'] = (data['Date'] - data['Date'].min()).dt.days  # Convert timedelta to days
    data['DaysAfterHoliday'] = (data['Date'] - data['Date'].max()).dt.days  # Convert timedelta to days
    data['MonthBegin'] = data['Date'].dt.day == 1
    data['MonthMid'] = data['Date'].dt.day >= 15
    data['MonthEnd'] = data['Date'].dt.day >= 25

    # Handle missing values
    data.fillna(0, inplace=True)
    logger.info("Handling missing values.")

    # Scale numeric features
    logger.info("Scaling numeric features.")

    scaler = StandardScaler()
    numeric_features = ['Sales', 'CompetitionDistance', 'DayOfWeek', 'DaysToNearestHoliday', 'DaysAfterHoliday']
    data[numeric_features] = scaler.fit_transform(data[numeric_features])
    logger.info("Preprocessing completed.")

    return data
    
# rossmann_sales_forecast_pipeline.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime



# Function to load data
def load_data(train_path):
    train = pd.read_csv(train_path)
    logging.info('Data loaded successfully')
    return train

# Feature engineering
def feature_engineering(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    df['IsBeginningOfMonth'] = df['Day'] <= 10
    df['IsEndOfMonth'] = df['Day'] >= 20
    logging.info('Feature engineering completed for dataset of shape %s', df.shape)
    return df

def encode_categorical(X_train, X_test):
    categorical_columns = ['StateHoliday']
    
    # Ensure both X_train and X_test are DataFrames
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError(f"X_train must be a DataFrame, but got {type(X_train)}")
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(f"X_test must be a DataFrame, but got {type(X_test)}")
    
    # Combine train and test for consistent encoding
    combined = pd.concat([X_train, X_test], axis=0)
    
    # Perform one-hot encoding
    combined_encoded = pd.get_dummies(combined, columns=categorical_columns, drop_first=True)
    
    # Split back to train and test based on the shapes of the original DataFrames
    X_train_encoded = combined_encoded.iloc[:X_train.shape[0], :].copy()
    X_test_encoded = combined_encoded.iloc[X_train.shape[0]:, :].copy()
    
    logging.info('Categorical encoding completed with columns: %s', list(X_train_encoded.columns))
    return X_train_encoded, X_test_encoded


# Handle missing values
def handle_missing(df):
    df.fillna(0, inplace=True)
    logging.info('Missing values handled')
    return df

# Train model
def train_model(X_train, y_train):
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    logging.info('Model training completed')
    return pipeline

# Evaluate model
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    logging.info(f'Model evaluation: RMSE = {rmse}')
    print(f'RMSE: {rmse}')
