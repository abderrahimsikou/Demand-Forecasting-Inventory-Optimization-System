import pandas as pd
import os
from loader import load_csv


def clean_data(df):
    """
    Clean the dataframe by converting Date column to datetime.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame with Date as datetime type
    """
    df = df.copy()
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # Save cleaned CSV to data/cleaned.csv
    cleaned_path = os.path.join(os.path.dirname(__file__), '../../data/cleaned.csv')
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned data saved to: {cleaned_path}")
    
    return df


def main():
    """Load, clean, and test the dataset."""
    print("\n" + "="*60)
    print("Data Cleaning Pipeline")
    print("="*60)
    
    print("\nLoading raw data...")
    df = load_csv()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    print("\nBefore cleaning:")
    print(f"Date column type: {df['Date'].dtype}")
    print(f"Sample dates: {df['Date'].head(3).tolist()}")
    
    print("\nCleaning data...")
    cleaned_df = clean_data(df)
    
    print("\nAfter cleaning:")
    print(f"Date column type: {cleaned_df['Date'].dtype}")
    print(f"Sample dates: {cleaned_df['Date'].head(3).tolist()}")
    
    print("\n" + "="*60)
    print("Cleaning complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
