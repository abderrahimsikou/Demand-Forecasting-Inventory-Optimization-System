from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError('XGBoost is required to train the baseline model. Install xgboost and rerun.') from exc

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except ImportError as exc:
    raise ImportError('Statsmodels is required to train the ARIMA baseline model. Install statsmodels and rerun.') from exc


class ARIMARegressor:
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.results_ = None

    def fit(self, X, y):
        self.results_ = SARIMAX(
            endog=y,
            exog=X,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, X):
        if self.results_ is None:
            raise ValueError('ARIMA model must be fitted before calling predict.')
        return self.results_.forecast(steps=len(X), exog=X)


def _coerce_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    coerced = pd.to_numeric(series, errors='coerce')
    if not coerced.isna().all():
        return coerced

    lower = series.dropna().astype(str).str.lower()
    if lower.isin({'true', 'false', 'yes', 'no', 'y', 'n'}).all():
        return lower.map({'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0}).astype(float)

    codes, uniques = pd.factorize(series, sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Could not find features.csv at {path}')
    df = pd.read_csv(path, low_memory=False)
    if 'Date' not in df.columns:
        raise ValueError("features.csv must contain a 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    if df['Date'].isna().all():
        raise ValueError('Unable to parse any Date values from features.csv')
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    return df


def prepare_split(df: pd.DataFrame):
    drop_columns = ['Units Sold', 'Store ID', 'Product ID', 'discount_amount', 'Date']
    missing_target = 'Units Sold' not in df.columns
    if missing_target:
        raise ValueError("features.csv must contain target column 'Units Sold'.")

    y = pd.to_numeric(df['Units Sold'], errors='coerce')
    if y.isna().all():
        raise ValueError("Target column 'Units Sold' contains no numeric values.")

    feature_columns = [col for col in df.columns if col not in drop_columns]
    X = pd.DataFrame({col: _coerce_column(df[col]) for col in feature_columns})

    date_values = df['Date'].dropna().unique()
    if len(date_values) < 2:
        raise ValueError('Need at least two unique dates to create a chronological split.')

    cutoff_index = max(1, int(len(date_values) * 0.8))
    cutoff_date = sorted(date_values)[cutoff_index - 1]

    train_mask = df['Date'] <= cutoff_date
    test_mask = df['Date'] > cutoff_date
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError('Chronological split produced an empty train or test set.')

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
        'KNeighbors': KNeighborsRegressor(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'ARIMA': ARIMARegressor(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_models(models, X_test, y_test):
    records = []
    for name, model in models.items():
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f'{name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}')
        records.append({'model': name, 'mae': mae, 'rmse': rmse, 'r2': r2})
    comparison = pd.DataFrame(records).sort_values(['rmse', 'mae']).reset_index(drop=True)
    print('\nComparison table:')
    print(comparison.to_string(index=False))
    return comparison


def save_best_model(models, comparison: pd.DataFrame, output_path: Path):
    best_name = comparison.loc[0, 'model']
    best_model = models[best_name]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, output_path)
    print(f'Saved best model ({best_name}) to {output_path}')


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'data' / 'features.csv'
    model_path = repo_root / 'models' / 'baseline.pkl'

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_split(df)
    models = train_models(X_train, y_train)
    comparison = evaluate_models(models, X_test, y_test)
    save_best_model(models, comparison, model_path)


if __name__ == '__main__':
    main()
