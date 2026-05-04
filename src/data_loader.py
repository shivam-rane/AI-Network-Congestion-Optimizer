import os

import pandas as pd


def load_data():
    data_path = "data/raw"

    all_files = [
        os.path.join(data_path, f)
        for f in os.listdir(data_path)
        if f.endswith(".csv")
    ]

    if len(all_files) == 0:
        raise FileNotFoundError("No CSV files found in data/raw")

    dfs = []

    for file in all_files:
        df = pd.read_csv(
            file,
            low_memory=False,
        )

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.columns = full_df.columns.str.strip()

    return full_df
