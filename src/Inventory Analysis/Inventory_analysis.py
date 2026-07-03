import os

# Create the directory if it doesn't exist
os.makedirs('src/Inventory Analysis', exist_ok=True)

import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(file_path):
   
    try:
        df = pd.read_csv(file_path)
        logging.info(f'Successfully loaded {file_path}')

        print("Data Head")
        print(df.head())
        print("Missing Values")
        print(df.isnull().sum())
        print("Data Info")
        print(df.info())
        print("Descriptive Statistics")
        print(df.describe())

        return df
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return None

def process_inventory(df):
    # 3. Create Inventory Gap
    df['Inventory Gap'] = df['Inventory Level'] - df['Demand Forecast']

    # 4. Create Days of Inventory
    # Use np.inf or 0 handling for division by zero if Demand Forecast is 0
    df['Days of Inventory'] = df['Inventory Level'] / df['Demand Forecast'].replace(0, np.nan)
    df['Days of Inventory'] = df['Days of Inventory'].map(lambda x: f'{x:.1f}' if pd.notna(x) else x)

    # 5. Classify Inventory Health (5% tolerance)
    def classify_health(row):
        inv = row['Inventory Level']
        demand = row['Demand Forecast']
        if inv < demand * 0.95:
            return 'Stockout Risk'
        elif inv > demand * 1.05:
            return 'Overstock'
        else:
            return 'Healthy'

    df['Inventory Status'] = df.apply(classify_health, axis=1)

    # 6. Create Suggested Order
    df['Suggested Order'] = np.where(
        df['Inventory Status'] == 'Stockout Risk',
        (df['Demand Forecast'] - df['Inventory Level']).clip(lower=0),
        0
    )

    return df

def clean_and_save(df, output_path):
    cols_to_keep = [
        'Product ID', 'Category', 'Inventory Level', 'Units Sold',
        'Price', 'Discount', 'Demand Forecast', 'Inventory Gap',
        'Days of Inventory', 'Inventory Status', 'Suggested Order',
        'Competitor Pricing'
    ]

    # Filter only columns that exist in the dataframe
    final_df = df[[c for c in cols_to_keep if c in df.columns]]

    final_df.to_csv(output_path, index=False)
    logging.info(f'Final dataset saved to {output_path}')

if __name__ == '__main__':
    input_file = 'data/prediction_data.csv'
    output_file = 'data/recommendation.csv'

    # Execution flow
    data = load_data(input_file)
    if data is not None:
        processed_data = process_inventory(data)
        clean_and_save(processed_data, output_file)

print('Script created at: src/Inventory Analysis/Inventory_Analysis.py')