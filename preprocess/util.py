import warnings

import numpy as np
import pandas as pd


def count_nonzero_ratio(l):
    return float(np.count_nonzero(l)) / float(len(l))


def count_nonzero(l):
    return np.count_nonzero(l)


def all_same(items):
    return all(x == items[0] for x in items)


def impute_missing_values(df, option="ffill"):
    # print("Raw dataframe:", df.shape)
    # flag indicators
    df_flags = df.isna()

    if option == "ffill":
        # forward fill with last valid observations
        df.fillna(method='ffill', inplace=True)

    # keep lines when all signals have valid values
    df.dropna(inplace=True)
    df_flags = df_flags.iloc[df.index]

    df.reset_index(drop=True, inplace=True)
    df_flags.reset_index(drop=True, inplace=True)

    assert df.shape == df_flags.shape
    assert df.isna().sum().sum() == 0

    # print("Dataframe after filling NaN:", df.shape)

    # col_name, % of non-NaN values
    # for col in df_flags.columns:
    #     print("{}: {:.2f} ({}/{})".format(col, 1-count_nonzero_ratio(df_flags[col]), len(
    #         df_flags[col])-count_nonzero(df_flags[col]), len(df_flags[col])))

    # for col in df.columns:
    #     if all_same(df[col]):
    #         warnings.warn("Column {} is constant!".format(col))

    return df, df_flags
