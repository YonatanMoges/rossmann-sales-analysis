import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy as scipy
import numpy as np


# Clean and prepare data
def clean_data(data):
    """Cleans and prepares the data."""
    # Handle missing values
    data.fillna(0, inplace=True)

    # Convert date columns
    data['Date'] = pd.to_datetime(data['Date'])

    # Create additional features
    data['Hour'] = data['Date'].dt.hour
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
    # Filter data for holidays
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

def analyze_seasonal_purchase_behaviors(data):
    """Analyzes seasonal purchase behaviors."""
    seasonal_sales = data.groupby('Month')['Sales'].mean()

    # Plot the results
    plt.plot(seasonal_sales.index, seasonal_sales.values)
    plt.xlabel('Month')
    plt.ylabel('Average Sales')
    plt.title('Seasonal Purchase Behaviors')
    plt.show()

    logging.info("Seasonal purchase behaviors analyzed.")

def analyze_correlation(data):
    """Analyzes the correlation between sales and the number of customers."""
    correlation = data['Sales'].corr(data['Customers'])

    print("Correlation between Sales and Customers:", correlation)

    logging.info("Correlation between Sales and Customers analyzed.")

def analyze_promo_impact(data):
    """Analyzes the impact of promotions on sales."""
    promo_sales = data.groupby('Promo')['Sales'].mean()

    # Plot the results
    plt.bar(promo_sales.index, promo_sales.values)
    plt.xlabel('Promo')
    plt.ylabel('Average Sales')
    plt.title('Impact of Promotions on Sales')
    plt.show()

    logging.info("Promotion impact analyzed.")

def analyze_customer_behavior(data):
    """Analyzes customer behavior during store opening and closing times."""
    hourly_sales = data.groupby(['DayOfWeek', 'Hour'])['Sales'].mean().reset_index()

    # Create a heatmap
    sns.heatmap(hourly_sales.pivot_table(index='Hour', columns='DayOfWeek', values='Sales'))
    plt.title('Hourly Sales by Day of Week')
    plt.show()

    logging.info("Customer behavior analyzed.")

def analyze_store_opening_hours(data):
    """Analyzes the impact of store opening hours on weekend sales."""
    # Identify stores open on all weekdays
    stores_open_all_weekdays = data.groupby('Store')['Open'].apply(lambda x: all(x == 1))

    # Filter data for weekend sales and store groups
    weekend_data = data[(data['DayOfWeek'] == 5) | (data['DayOfWeek'] == 6)]  # Use specific day numbers for Saturday and Sunday
    stores_all_weekdays_data = weekend_data[weekend_data['Store'].isin(stores_open_all_weekdays[stores_open_all_weekdays == True].index)]
    stores_fewer_weekdays_data = weekend_data[~weekend_data['Store'].isin(stores_open_all_weekdays[stores_open_all_weekdays == True].index)]

    # Check for empty groups
    if stores_all_weekdays_data.empty or stores_fewer_weekdays_data.empty:
        print("One or both groups are empty. Check filtering criteria and data consistency.")

        # Further analysis
        print("Stores open all weekdays:", stores_open_all_weekdays.sum())
        print("Weekend data shape:", weekend_data.shape)
        print("Open column values:", data['Open'].value_counts())
        print("DayOfWeek column values:", data['DayOfWeek'].value_counts())

    else:
        # Calculate average weekend sales for each group
        avg_weekend_sales_all_weekdays = stores_all_weekdays_data['Sales'].mean()
        avg_weekend_sales_fewer_weekdays = stores_fewer_weekdays_data['Sales'].mean()

        # Plot the results
        plt.bar(['All Weekdays', 'Fewer Weekdays'], [avg_weekend_sales_all_weekdays, avg_weekend_sales_fewer_weekdays])
        plt.xlabel('Store Opening Schedule')
        plt.ylabel('Average Weekend Sales')
        plt.title('Impact of Store Opening Hours on Weekend Sales')
        plt.show()

    logging.info("Store opening hours analyzed.")

def analyze_assortment_impact(data):
    """Analyzes the impact of assortment type on sales."""
    assortment_sales = data.groupby('Assortment')['Sales'].mean()

    # Plot the results
    plt.bar(assortment_sales.index, assortment_sales.values)
    plt.xlabel('Assortment')
    plt.ylabel('Average Sales')
    plt.title('Impact of Assortment Type on Sales')
    plt.show()

    logging.info("Assortment impact analyzed.")

def analyze_competitor_distance(data):
    """Analyzes the impact of competitor distance on sales."""
    # Group by 'CompetitionDistance' and calculate average sales
    distance_sales = data.groupby('CompetitionDistance')['Sales'].mean()

    # Plot the results
    plt.plot(distance_sales.index, distance_sales.values)
    plt.xlabel('Competition Distance')
    plt.ylabel('Average Sales')
    plt.title('Impact of Competitor Distance on Sales')
    plt.show()

def analyze_new_competitors(data):
    """Analyzes the impact of new competitors on sales."""

    # Use all stores for the analysis
    stores_with_new_competitors = data.dropna(subset=['Sales', 'Date', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth'])

    # Ensure 'Date' is in datetime format
    stores_with_new_competitors['Date'] = pd.to_datetime(stores_with_new_competitors['Date'])

    # Create a combined date column for competition opening
    stores_with_new_competitors['CompetitionOpenSince'] = pd.to_datetime(
        stores_with_new_competitors['CompetitionOpenSinceYear'].astype(int).astype(str) + '-' +
        stores_with_new_competitors['CompetitionOpenSinceMonth'].astype(int).astype(str), format='%Y-%m', errors='coerce'
    )

    # Calculate time since competitor opening
    stores_with_new_competitors['TimeSinceCompetitorOpening'] = stores_with_new_competitors['Date'] - stores_with_new_competitors['CompetitionOpenSince']

    # Filter out rows where competition opening date is after the sales date
    stores_with_new_competitors = stores_with_new_competitors.dropna(subset=['CompetitionOpenSince', 'TimeSinceCompetitorOpening'])

    # Group by store, store type, and assortment, and calculate sales differences
    def sales_difference(group):
        pre_competition_sales = group.loc[group['TimeSinceCompetitorOpening'] < pd.Timedelta(days=0), 'Sales']
        post_competition_sales = group.loc[group['TimeSinceCompetitorOpening'] >= pd.Timedelta(days=0), 'Sales']
        if len(pre_competition_sales) > 0 and len(post_competition_sales) > 0:
            return pre_competition_sales.mean() - post_competition_sales.mean()
        else:
            return np.nan  # Return NaN if no pre or post data

    grouped_data = stores_with_new_competitors.groupby(['Store']).apply(sales_difference).dropna()

    # Ensure we have data for t-test
    if grouped_data.empty:
        print("No valid data available for t-test.")
        return

    # Perform t-test
    t_statistic, p_value = scipy.stats.ttest_1samp(grouped_data, 0)

    # Print results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped_data)
    plt.xlabel('Store Group')
    plt.ylabel('Sales Difference')
    plt.title('Impact of New Competitors on Sales')
    plt.show()


