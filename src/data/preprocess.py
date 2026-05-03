"""Clean raw CICIDS data."""

import numpy as np
import pandas as pd

from src.utils.config import RAW_FEATURES


def clean_raw_data(raw_df):
    """Convert selected columns to numeric and remove invalid rows."""
    df = raw_df.copy()

    for column in RAW_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=RAW_FEATURES)
    df = df[df["Flow Duration"] > 0]
    df = df[df["Flow Bytes/s"] >= 0]
    df = df[df["Flow Packets/s"] >= 0]

    return df.reset_index(drop=True)
