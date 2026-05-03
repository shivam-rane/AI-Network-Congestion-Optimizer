"""Load CICIDS CSV data efficiently."""

import pandas as pd

from src.utils.config import CICIDS_FILES, MAX_ROWS, RAW_FEATURES


def strip_column_names(columns):
    """Remove leading/trailing spaces from CICIDS column names."""
    return [column.strip() for column in columns]


def load_cicids_data(file_paths=None, max_rows=MAX_ROWS):
    """Load only the required CICIDS columns up to max_rows."""
    file_paths = file_paths or CICIDS_FILES
    data_parts = []
    loaded_rows = 0
    original_columns = 0

    for file_path in file_paths:
        if not file_path.exists():
            continue

        header = pd.read_csv(file_path, nrows=0)
        original_columns = max(original_columns, len(header.columns))
        column_map = {column.strip(): column for column in header.columns}
        usecols = [column_map[column] for column in RAW_FEATURES if column in column_map]

        if len(usecols) != len(RAW_FEATURES):
            continue

        remaining_rows = max_rows - loaded_rows
        if remaining_rows <= 0:
            break

        df_part = pd.read_csv(file_path, usecols=usecols, nrows=remaining_rows)
        df_part.columns = strip_column_names(df_part.columns)
        data_parts.append(df_part)
        loaded_rows += len(df_part)

    if not data_parts:
        raise FileNotFoundError("No CICIDS CSV files could be loaded. Check config.CICIDS_FILES.")

    raw_df = pd.concat(data_parts, ignore_index=True)
    return raw_df, loaded_rows, original_columns


if __name__ == "__main__":
    df, rows, cols = load_cicids_data()
    print(f"Loaded rows: {rows}")
    print(f"Original columns: {cols}")
    print(df.head())
