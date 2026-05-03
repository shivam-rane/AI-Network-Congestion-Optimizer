"""Preprocessing for raw CICIDS network flow data."""

import numpy as np
import pandas as pd

from src.config import RAW_COLUMNS


def clean_raw_data(raw_df):
    """Convert raw CICIDS fields to numeric, remove invalid rows, and filter outliers."""
    df = raw_df.copy()

    for column in RAW_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=RAW_COLUMNS)
    df = df[df["Flow Duration"] > 0]
    df = df[df["Flow Bytes/s"] >= 0]
    df = df[df["Flow Packets/s"] >= 0]

    # Filter extreme outliers using robust upper quantiles.
    for column in RAW_COLUMNS:
        upper = df[column].quantile(0.995)
        df = df[df[column] <= upper]

    return df.reset_index(drop=True)


def robust_minmax(series):
    """Scale a series to 0-1 using the 1st and 99th percentiles."""
    lower = series.quantile(0.01)
    upper = series.quantile(0.99)
    if upper <= lower:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - lower) / (upper - lower)).clip(0, 1)
