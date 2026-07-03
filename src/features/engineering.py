import pandas as pd
import numpy as np


def create_features(df):
    """
    Engineer features for demand forecasting with business-driven rationale.

    Args:
        df: DataFrame with columns including Date, Category, Seasonality, Store ID, Product ID,
            Units Sold, Price, Inventory Level, Units Ordered, Discount, 
            Competitor Pricing, Holiday/Promotion, Region, Weather Condition

    Returns:
        DataFrame with engineered features. Columns that leak future information are dropped
        from the returned feature set (kept in the intermediate dataframe for analysis if needed).
    """
    df = df.copy()

    # ============================================================================
    # 1. TEMPORAL FEATURES - Extract time-based patterns that affect demand
    # ============================================================================
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = ((df['Date'].dt.dayofweek >= 5) & (df['Date'].dt.dayofweek <= 6)).astype(int)
        df['quarter'] = df['Date'].dt.quarter
        df['day_of_month'] = df['Date'].dt.day
    else:
        # create placeholders if Date missing
        df['month'] = df.get('month', 0)
        df['day_of_week'] = df.get('day_of_week', 0)
        df['is_weekend'] = df.get('is_weekend', 0)
        df['quarter'] = df.get('quarter', 0)
        df['day_of_month'] = df.get('day_of_month', 0)

    # ============================================================================
    # 2. CATEGORICAL ENCODING
    # ============================================================================
    if 'Category' in df.columns:
        category_dummies = pd.get_dummies(df['Category'], prefix='cat_')
        df = pd.concat([df, category_dummies], axis=1)

    seasonality_map = {'Spring': 0, 'Summer': 1, 'Autumn': 2, 'Winter': 3}
    if 'Seasonality' in df.columns:
        df['seasonality_encoded'] = df['Seasonality'].map(seasonality_map)
    else:
        df['seasonality_encoded'] = df.get('seasonality_encoded', 0)

    # ============================================================================
    # 3. ROLLING STATISTICS & LAGS - computed grouped by Store ID and Product ID
    # These features are based only on prior observations and do not leak current-day demand.
    # ============================================================================
    if ('Store ID' in df.columns) and ('Product ID' in df.columns) and ('Units Sold' in df.columns):
        sort_cols = ['Store ID', 'Product ID', 'Date']
        sorted_df = df.sort_values(sort_cols, na_position='last').copy()
        grouped = sorted_df.groupby(['Store ID', 'Product ID'], sort=False)
        sorted_df['units_sold_lag1'] = grouped['Units Sold'].shift(1)
        sorted_df['units_sold_lag7'] = grouped['Units Sold'].shift(7)
        sorted_df['units_sold_7d_avg'] = grouped['Units Sold'].shift(1).rolling(window=7, min_periods=1).mean()
        sorted_df['units_sold_14d_avg'] = grouped['Units Sold'].shift(1).rolling(window=14, min_periods=1).mean()
        df[['units_sold_lag1', 'units_sold_lag7', 'units_sold_7d_avg', 'units_sold_14d_avg']] = \
            sorted_df[['units_sold_lag1', 'units_sold_lag7', 'units_sold_7d_avg', 'units_sold_14d_avg']]
    else:
        for col in ['units_sold_7d_avg', 'units_sold_14d_avg', 'units_sold_lag1', 'units_sold_lag7']:
            df[col] = df.get(col, np.nan)

    # ============================================================================
    # 4. DOMAIN-SPECIFIC FEATURES
    # ============================================================================
    df['discount_amount'] = df.get('Price', 0) * (df.get('Discount', 0) / 100)
    df['price_advantage'] = df.get('Competitor Pricing', 0) - df.get('Price', 0)

    # Finalize feature set.
    cols_to_drop = ['Category', 'Region', 'Weather Condition', 'Seasonality', 'Units Ordered']
    feature_df = df.copy()
    feature_df = feature_df.drop(columns=cols_to_drop, errors='ignore')

    # Convert object-like columns to numeric where possible (booleans, numeric strings)
    for col in feature_df.columns:
        if feature_df[col].dtype == object:
            vals = feature_df[col].dropna().astype(str).str.lower().unique()[:10]
            if set(vals).issubset({'true', 'false'}):
                feature_df[col] = feature_df[col].map(lambda v: 1 if str(v).lower() == 'true' else 0)
            else:
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_df[num_cols] = feature_df[num_cols].fillna(0)

    non_numeric = [c for c in feature_df.columns if feature_df[c].dtype == object]
    if non_numeric:
        feature_df = feature_df.drop(columns=non_numeric)

    return feature_df


def select_features(df, variance_threshold=None):
    """
    Selects features by removing highly correlated.
    """
    df = df.copy()
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute the absolute correlation matrix for numeric features
    corr_matrix = numeric_df.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    correlated_drops = []
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > 0.95):
            correlated_drops.append(column)

    if correlated_drops:
        print(f"Dropping correlated features (>0.95): {correlated_drops}")
    else:
        print("No highly correlated features found.")

    reduced_df = df.drop(columns=correlated_drops, errors='ignore')

    selected_features = reduced_df.columns.tolist()
    return selected_features, reduced_df


if __name__ == '__main__':
    # Load cleaned data
    df = pd.read_csv('data/cleaned.csv')
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Create features
    df_features = create_features(df)
    print(f"\nFeatures shape: {df_features.shape}")
    print(f"\nNew engineered features:")
    print(df_features.columns.tolist())
    
    # Save features
    df_features.to_csv('data/features.csv', index=False)
    print(f"\nFeatures saved to 'data/features.csv'")
    
    # Display first 5 rows
    print(f"\nFirst 5 rows of engineered features:")
    print(df_features.head())
