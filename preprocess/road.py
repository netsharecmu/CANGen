import pandas as pd

from .util import impute_missing_values


def preprocess_single_road_csv(csv_file):
    df = pd.read_csv(csv_file)
    raw_columns = df.columns
    # 1. Drop signal columns that are all NaN
    df.dropna(axis=1, how='all', inplace=True)

    # print removed columns
    removed_columns = set(raw_columns) - set(df.columns)
    print("Removed columns:", removed_columns)

    # 2. Impute missing values with `ffill` (i.e., forward fill)
