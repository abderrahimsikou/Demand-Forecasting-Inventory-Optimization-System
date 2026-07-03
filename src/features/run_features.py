import os
import sys
import time
import pandas as pd

# Ensure src directory is importable as a package root
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from features.engineering import create_features, select_features


def main():
    t0 = time.time()

    project_root = os.path.abspath(os.path.join(src_dir, '..'))
    data_in = os.path.join(project_root, 'data', 'cleaned.csv')
    data_out = os.path.join(project_root, 'data', 'features.csv')

    print(f"Loading data from: {data_in}")
    df = pd.read_csv(data_in)
    print(f"Original shape: {df.shape}")

    # Create features
    df_feats = create_features(df)
    print(f"After feature engineering shape: {df_feats.shape}")

    # Select features
    selected_features, reduced_df = select_features(df_feats)
    print(f"After feature selection shape: {reduced_df.shape}")
    print(f"Kept features ({len(selected_features)}): {selected_features}")

    # Save to CSV
    os.makedirs(os.path.dirname(data_out), exist_ok=True)
    reduced_df.to_csv(data_out, index=False)
    elapsed = time.time() - t0
    print(f"Saved engineered features to: {data_out}")
    print(f"Elapsed time: {elapsed:.2f}s")


if __name__ == '__main__':
    main()
