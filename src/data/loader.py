import pandas as pd
import os


def load_csv(filename='retail_store_inventory.csv'):
    """Load CSV file from data folder."""
    data_path = os.path.join(os.path.dirname(__file__), '../../data', filename)
    df = pd.read_csv(data_path)
    return df


def print_shape(df):
    """Print dataset shape (rows, columns)."""
    rows, cols = df.shape
    print(f"\n{'='*60}")
    print(f"Dataset Shape: {rows} rows, {cols} columns")
    print(f"{'='*60}")


def print_column_info(df):
    """Print column names and data types."""
    print(f"\n{'='*60}")
    print("Column Info:")
    print(f"{'='*60}")
    df.info()
    print(f"\n{'='*60}")
    print("Data Types:")
    print(f"{'='*60}")
    print(df.dtypes)


def print_summary_statistics(df):
    """Print summary statistics for numeric columns."""
    print(f"\n{'='*60}")
    print("Summary Statistics (Numeric Columns):")
    print(f"{'='*60}")
    numeric_df = df.describe()
    print(numeric_df.to_string())


def print_missing_duplicates(df):
    """Print missing values and duplicated row counts with percentages."""
    total_rows = len(df)
    
    print(f"\n{'='*60}")
    print("Missing Values:")
    print(f"{'='*60}")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        for col in df.columns:
            if missing[col] > 0:
                percentage = (missing[col] / total_rows) * 100
                print(f"{col:30s} | {missing[col]:6d} ({percentage:6.2f}%)")
    
    print(f"\n{'='*60}")
    print("Duplicated Values:")
    print(f"{'='*60}")
    duplicated_count = df.duplicated().sum()
    if duplicated_count == 0:
        print("No duplicated rows found.")
    else:
        percentage = (duplicated_count / total_rows) * 100
        print(f"Total duplicated rows: {duplicated_count} ({percentage:.2f}%)")


def main():
    """Load and analyze CSV data."""
    print("\nLoading CSV file from data/ folder...")
    df = load_csv()
    
    print_shape(df)
    print_column_info(df)
    print_summary_statistics(df)
    print_missing_duplicates(df)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()