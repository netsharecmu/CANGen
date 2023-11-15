import os

import pandas as pd

from util import impute_missing_values


def preprocess_single_road_csv(csv_file):
    df = pd.read_csv(csv_file)
    raw_columns = df.columns
    num_raw_rows = df.shape[0]
    # 1. Drop signal columns that are all NaN
    df.dropna(axis=1, how='all', inplace=True)

    # print removed columns
    removed_columns = set(raw_columns) - set(df.columns)
    print("Removed columns:", removed_columns)

    # 2. Impute missing values with `ffill` (i.e., forward fill)
    # print("Before imputation:", df.shape)
    df, _ = impute_missing_values(df)
    # print("After imputation:", df.shape)
    num_removed_rows = num_raw_rows - df.shape[0]
    print(
        f"{num_removed_rows}/{df.shape[0]} ({num_removed_rows/df.shape[0]*100:.2f}%) rows are removed")

    return df


if __name__ == "__main__":
    in_dir = "../data/road/signal_extractions"
    out_dir = "../data/road/signal_extractions_preprocessed"
    os.makedirs(out_dir, exist_ok=True)

    for sub_in_dir in ['ambient', 'attacks']:
        os.makedirs(os.path.join(out_dir, sub_in_dir), exist_ok=True)
        for csv_file in os.listdir(os.path.join(in_dir, sub_in_dir)):
            if csv_file.endswith(".csv"):
                print(csv_file)
                df = preprocess_single_road_csv(
                    os.path.join(in_dir, sub_in_dir, csv_file))
                df.to_csv(os.path.join(
                    out_dir, sub_in_dir, csv_file), index=False)
                print()
