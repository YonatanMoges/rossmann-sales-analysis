
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.utils import resample
import pickle
import time

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

def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'RMSE: {rmse}')


def plot_feature_importance(model, X_train):
    
    importances = model.named_steps['model'].feature_importances_
    feature_names = X_train.columns
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), feature_names[indices], rotation=90)
    plt.tight_layout()  # Adjust layout to ensure labels are readable
    plt.show()
    logging.info("Feature importance plot generated successfully.")

def estimate_confidence_intervals(X_train, y_train, X_test, n_iterations=100, alpha=0.95, n_jobs=-1):
    """
    Estimate prediction confidence intervals using bootstrapping and parallel processing.
    
    Args:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target values.
    X_test (pd.DataFrame): Test features for which to predict.
    n_iterations (int): Number of bootstrap iterations to perform.
    alpha (float): Confidence level. Default is 95%.
    n_jobs (int): The number of CPU cores to use (-1 means use all available cores).
    
    Returns:
    lower_bound (np.array): Lower bound of the confidence intervals.
    upper_bound (np.array): Upper bound of the confidence intervals.
    """
    def train_and_predict(i):
        X_resample, y_resample = resample(X_train, y_train, replace=True, random_state=i)
        model = Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        model.fit(X_resample, y_resample)
        return model.predict(X_test)

    # Parallelize the bootstrap iterations
    predictions = Parallel(n_jobs=n_jobs)(delayed(train_and_predict)(i) for i in range(n_iterations))

    # Convert predictions to a numpy array
    predictions = np.array(predictions)

    # Calculate the lower and upper percentiles
    lower_bound = np.percentile(predictions, (1 - alpha) / 2 * 100, axis=0)
    upper_bound = np.percentile(predictions, (alpha + (1 - alpha) / 2) * 100, axis=0)
    logging.info(f"Bootstrap confidence intervals estimated. Lower Bound: {lower_bound}, Upper Bound: {upper_bound}")
    return lower_bound, upper_bound

def serialize_model(model):
    # Get the current timestamp in the format "dd-mm-yyyy-HH-MM-SS"
    timestamp = time.strftime("%d-%m-%Y-%H-%M-%S")
    model_filename = f"../models/sales_model_{timestamp}.pkl"
    
    # Serialize the model to a file
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    
    logging.info(f'Model saved as {model_filename}')
    print(f'Model saved as {model_filename}')
    


