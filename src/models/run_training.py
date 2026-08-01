import json
import os
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


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


def compute_metrics(y_true, y_pred):
    """Compute MAE, RMSE, R² metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def train_baseline(X_train, X_test, y_train, y_test):
    """Train baseline models and return the best one."""
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'KNeighbors': KNeighborsRegressor(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    
    if XGBRegressor is not None:
        models['XGBoost'] = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

    best_model = None
    best_name = None
    best_test_rmse = float('inf')
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        
        train_metrics = compute_metrics(y_train, train_preds)
        test_metrics = compute_metrics(y_test, test_preds)
        
        results[name] = {
            'model': model,
            'train': train_metrics,
            'test': test_metrics
        }
        
        if test_metrics['rmse'] < best_test_rmse:
            best_test_rmse = test_metrics['rmse']
            best_model = model
            best_name = name

    return best_model, best_name, results


def train_tuned(X_train, X_test, y_train, y_test, best_params_path: Path):
    """Train tuned GradientBoosting model with best hyperparameters."""
    if not best_params_path.exists():
        raise FileNotFoundError(f'Best params not found at {best_params_path}')
    
    with open(best_params_path) as f:
        best_params = json.load(f)
    
    model = GradientBoostingRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_metrics = compute_metrics(y_train, train_preds)
    test_metrics = compute_metrics(y_test, test_preds)
    
    return model, best_params, train_metrics, test_metrics


def log_mlflow_run(model_name, model, train_metrics, test_metrics, hyperparams, model_save_dir: Path):
    """Log a single model run to MLflow."""
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        mlflow.log_param('model_name', model_name)
        if hyperparams:
            for key, value in hyperparams.items():
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
        
        # Log train metrics
        mlflow.log_metric('train_mae', train_metrics['mae'])
        mlflow.log_metric('train_rmse', train_metrics['rmse'])
        mlflow.log_metric('train_r2', train_metrics['r2'])
        
        # Log test metrics
        mlflow.log_metric('test_mae', test_metrics['mae'])
        mlflow.log_metric('test_rmse', test_metrics['rmse'])
        mlflow.log_metric('test_r2', test_metrics['r2'])
        
        # Log model artifact
        temp_model_path = model_save_dir / f'{model_name}_temp.pkl'
        joblib.dump(model, temp_model_path)
        mlflow.log_artifact(str(temp_model_path), artifact_path='models')
        temp_model_path.unlink()
        
        print(f"Logged {model_name}:")
        print(f"  Train - MAE: {train_metrics['mae']:.4f}, RMSE: {train_metrics['rmse']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"  Test  - MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}, R²: {test_metrics['r2']:.4f}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'data' / 'features.csv'
    models_dir = repo_root / 'models'
    best_params_path = models_dir / 'best_params.json'
    production_model_path = models_dir / 'production_model.pkl'

    # Set MLflow tracking to SQLite database backend
    mlflow_db_path = repo_root / 'mlflow.db'
    mlflow.set_tracking_uri(f'sqlite:///{mlflow_db_path}')
    mlflow.set_experiment('demand_forecasting')

    print("Loading and preparing data...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_split(df)
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    models_dir.mkdir(parents=True, exist_ok=True)

    # Train and log baseline
    print("\n=== Training Baseline ===")
    baseline_model, best_baseline_name, baseline_results = train_baseline(X_train, X_test, y_train, y_test)
    baseline_train_metrics = baseline_results[best_baseline_name]['train']
    baseline_test_metrics = baseline_results[best_baseline_name]['test']
    log_mlflow_run('baseline', baseline_model, baseline_train_metrics, baseline_test_metrics, {}, models_dir)

    # Train and log tuned model
    print("\n=== Training Tuned Model ===")
    if best_params_path.exists():
        tuned_model, best_params, tuned_train_metrics, tuned_test_metrics = train_tuned(
            X_train, X_test, y_train, y_test, best_params_path
        )
        log_mlflow_run('tuned_best', tuned_model, tuned_train_metrics, tuned_test_metrics, best_params, models_dir)
    else:
        print(f"Warning: best_params.json not found at {best_params_path}")
        print("Skipping tuned model training. Run src/models/tuning.py first.")
        tuned_model = None

    # Save production model
    if tuned_model is not None:
        joblib.dump(tuned_model, production_model_path)
        print(f"\nSaved production model to {production_model_path}")
    else:
        joblib.dump(baseline_model, production_model_path)
        print(f"\nSaved baseline model as production model to {production_model_path}")

    print(f"\nMLflow tracking database: {repo_root / 'mlflow.db'}")
    print(f"Start MLflow server with: mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db")
    print(f"Then navigate to http://localhost:5000 to see all runs.")


if __name__ == '__main__':
    main()
