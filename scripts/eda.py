import logging
import matplotlib.pyplot as plt

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

