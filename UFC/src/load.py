import os
import pandas as pd


def save_processed(df, name, out_dir):
    """Write a DataFrame that was actually used in an analysis to data/processed."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(path, index=False)
    return path


def read_processed(name, in_dir):
    return pd.read_csv(os.path.join(in_dir, f"{name}.csv"))

