"""Load real CICIDS CSV files from the project data folder."""

from pathlib import Path

import pandas as pd

from src.config import MAX_ROWS, RAW_COLUMNS, RAW_DATA_DIR


def _strip_columns(columns):
    """Normalize CICIDS headers because many contain leading spaces."""
    return [column.strip() for column in columns]


def discover_cicids_files(data_dir=RAW_DATA_DIR):
    """Find real CICIDS CSV files under data/raw."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("*.csv"))
    return [path for path in files if path.is_file()]


def load_cicids_dataset(data_dir=RAW_DATA_DIR, max_rows=MAX_ROWS):
    """
    Load real CICIDS data from data/raw.

    Only required raw columns are loaded, keeping memory use predictable.
    """
    csv_files = discover_cicids_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(
            f"No CICIDS CSV files found in {Path(data_dir).resolve()}. "
            "Place the real CICIDS CSV files in data/raw/."
        )

    parts = []
    loaded_rows = 0
    original_columns = 0

    for csv_file in csv_files:
        header = pd.read_csv(csv_file, nrows=0)
        original_columns = max(original_columns, len(header.columns))
        column_map = {column.strip(): column for column in header.columns}
        usecols = [column_map[column] for column in RAW_COLUMNS if column in column_map]

        if len(usecols) != len(RAW_COLUMNS):
            continue

        remaining_rows = max_rows - loaded_rows
        if remaining_rows <= 0:
            break

        chunk = pd.read_csv(csv_file, usecols=usecols, nrows=remaining_rows)
        chunk.columns = _strip_columns(chunk.columns)
        parts.append(chunk)
        loaded_rows += len(chunk)

    if not parts:
        raise ValueError("CICIDS files were found, but required columns were missing.")

    return pd.concat(parts, ignore_index=True), loaded_rows, original_columns


if __name__ == "__main__":
    frame, rows, columns = load_cicids_dataset()
    print(f"Loaded rows: {rows}")
    print(f"Original columns: {columns}")
    print(frame.head())
