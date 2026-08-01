import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import optuna
except ImportError as exc:
    raise ImportError('Optuna is required for hyperparameter tuning. Install optuna and rerun.') from exc


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


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series):
    """Optuna objective function for hyperparameter tuning GradientBoosting."""
    n_estimators = trial.suggest_int('n_estimators', 50, 300, step=10)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42,
        n_iter_no_change=10,
        validation_fraction=0.1
    )

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    mean_score = scores.mean()

    trial.set_user_attr('cv_scores', scores.tolist())
    print(f"Trial {trial.number}: CV R² = {mean_score:.4f} (std: {scores.std():.4f})")

    return mean_score


def tune_hyperparameters(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 10):
    """Run Optuna optimization for GradientBoosting hyperparameters."""
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)

    best_trial = study.best_trial
    print(f"\nBest trial: Trial {best_trial.number}")
    print(f"Best CV R²: {best_trial.value:.4f}")
    print(f"Best hyperparameters: {best_trial.params}")

    trial_log = []
    for trial in study.trials:
        record = {
            'trial_number': trial.number,
            'hyperparameters': trial.params,
            'cv_score': trial.value,
            'cv_scores': trial.user_attrs.get('cv_scores', [])
        }
        trial_log.append(record)

    return best_trial.params, trial_log


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: dict):
    """Train final GradientBoosting model with best hyperparameters."""
    model = GradientBoostingRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate model on test set and print metrics."""
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\nFinal Model Metrics on Test Set:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def save_results(best_params: dict, trial_log: list, final_metrics: dict, model, output_dir: Path):
    """Save best params, trial log, and trained model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / 'best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved best hyperparameters to {params_path}")

    model_path = output_dir / 'tuned_model.pkl'
    joblib.dump(model, model_path)
    print(f"Saved tuned model to {model_path}")

    trial_log_path = output_dir / 'trial_log.json'
    with open(trial_log_path, 'w') as f:
        json.dump(trial_log, f, indent=2)
    print(f"Saved trial log to {trial_log_path}")


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / 'data' / 'features.csv'
    output_dir = repo_root / 'models'

    print("Loading and preparing data...")
    df = load_data(data_path)
    X_train, X_test, y_train, y_test = prepare_split(df)
    print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

    print("\nStarting hyperparameter tuning with Optuna (30 trials, 5-fold CV)...")
    best_params, trial_log = tune_hyperparameters(X_train, y_train, n_trials=30)

    print("\nTraining final model with best hyperparameters...")
    final_model = train_final_model(X_train, y_train, best_params)

    print("\nEvaluating on test set...")
    final_metrics = evaluate_model(final_model, X_test, y_test)

    print("\nSaving results...")
    save_results(best_params, trial_log, final_metrics, final_model, output_dir)

    print("\nTuning complete!")


if __name__ == '__main__':
    main()
