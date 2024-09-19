import logging
import matplotlib.pyplot as plt
import pandas as pd

# Clean and prepare data
def clean_data(data):
    """Cleans and prepares the data."""
    # Handle missing values
    data.fillna(0, inplace=True)

    # Convert date columns
    data['Date'] = pd.to_datetime(data['Date'])

    # Create additional features
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    return data

def analyze_promo_distribution(train_data, test_data):
    """Analyzes the distribution of promotions in training and test sets."""
    # Group by 'Promo' and count the number of observations
    train_promo_counts = train_data.groupby('Promo').size()
    test_promo_counts = test_data.groupby('Promo').size()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(train_promo_counts.index, train_promo_counts.values, label='Train')
    plt.bar(test_promo_counts.index, test_promo_counts.values, label='Test')
    plt.xlabel('Promo')
    plt.ylabel('Count')
    plt.title('Distribution of Promotions')
    plt.legend()
    plt.show()

    logging.info("Promotion distribution analyzed.")

def analyze_sales_behavior(data):
    """Analyzes sales behavior before, during, and after holidays."""
    holiday_data = data[data['StateHoliday'] != 0]

    # Calculate average sales for different periods
    avg_sales_before = holiday_data[holiday_data['Day'] < holiday_data['Day'] - 7]['Sales'].mean()
    avg_sales_during = holiday_data['Sales'].mean()
    avg_sales_after = holiday_data[holiday_data['Day'] > holiday_data['Day'] + 7]['Sales'].mean()

    # Plot the results
    plt.bar(['Before', 'During', 'After'], [avg_sales_before, avg_sales_during, avg_sales_after])
    plt.xlabel('Period')
    plt.ylabel('Average Sales')
    plt.title('Sales Behavior Before, During, and After Holidays')
    plt.show()

    logging.info("Sales behavior analyzed.")



