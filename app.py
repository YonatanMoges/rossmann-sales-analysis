# app.py

from flask import Flask, request, jsonify
import pandas as pd
import pickle
import logging
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress the warning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize the Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the serialized machine learning model
model_path = './models/sales_model_25-09-2024-18-53-49.pkl'  # Replace <timestamp> with the actual timestamp of your saved model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Preprocess function to handle incoming data (modify as necessary based on your feature engineering steps)
def preprocess_input(data):
    # Create a DataFrame from input data
    df = pd.DataFrame(data, index=[0])

    # Feature engineering (same as training)
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
    df['IsWeekend'] = df['DayOfWeek'] >= 5
    df['IsBeginningOfMonth'] = df['Day'] <= 10
    df['IsEndOfMonth'] = df['Day'] >= 20

    # Fill missing values with default values
    df['Promo'] = df.get('Promo', 0)
    df['SchoolHoliday'] = df.get('SchoolHoliday', 0)

    # Encode StateHoliday (assuming the training used one-hot encoding)
    df['StateHoliday'] = df.get('StateHoliday', '0')  # Default to '0'
    df_encoded = pd.get_dummies(df, columns=['StateHoliday'], drop_first=True)

    # Handle missing columns in case some categories are not present
    expected_columns = ['Year', 'Month', 'Day', 'DayOfWeek', 'IsWeekend',
                        'IsBeginningOfMonth', 'IsEndOfMonth', 'Promo',
                        'SchoolHoliday', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c', 'Id', 'Open']

    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Default value for missing columns

    # Ensure the order of columns is the same as in the training data
    df_encoded = df_encoded[expected_columns]

    return df_encoded


# Define the prediction endpoint
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Parse input data
        input_data = request.json

        # Check for required fields (e.g., Date)
        if 'Date' not in input_data:
            return jsonify({"error": "Missing 'Date' field", "message": "Please include 'Date' in the input."}), 400

        # Preprocess the input data
        df = preprocess_input(input_data)

        # Predict using the model
        prediction = model.predict(df)

        # Return the prediction in JSON format
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e), "message": "Error occurred during prediction"}), 500


@app.route('/')
def index():
    return "Welcome to the API"

if __name__ == "__main__":
    app.run(debug=True)



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
