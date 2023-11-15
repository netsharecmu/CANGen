import json
import os
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm


def count_nonzero_ratio(l):
    return float(np.count_nonzero(l)) / float(len(l))


def count_nonzero(l):
    return np.count_nonzero(l)


def all_same(items):
    return all(x == items[0] for x in items)


def impute_missing_values(df, option="ffill"):
    print("Raw dataframe:", df.shape)
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

    print("Dataframe after filling NaN:", df.shape)

    # col_name, % of non-NaN values
    for col in df_flags.columns:
        print("{}: {:.2f} ({}/{})".format(col, 1-count_nonzero_ratio(df_flags[col]), len(
            df_flags[col])-count_nonzero(df_flags[col]), len(df_flags[col])))

    for col in df.columns:
        if all_same(df[col]):
            warnings.warn("Column {} is constant!".format(col))

    return df, df_flags


def openxc_json2csv(
    in_json_file,
    out_csv_file,
    selected_cols=[
        "timestamp",
        'brake_pedal_status',
        'accelerator_pedal_position',
        'transmission_gear_position',
        'vehicle_speed',
        'engine_speed',
    ]
):
    data = []
    with open(in_json_file) as f:
        for line in f:
            data.append(json.loads(line))

    columns = []
    for row in tqdm(data):
        if row["name"] not in columns:
            columns.append(row["name"])
    # print(columns)

    dict_list = {}
    for row in tqdm(data):
        if row["timestamp"] not in dict_list:
            dict_list[row["timestamp"]] = {}
            for col in columns:
                dict_list[row["timestamp"]][col] = np.nan

        dict_list[row["timestamp"]][row["name"]] = row["value"]

    pd_dict_list = []
    for timestamp, kvs in dict_list.items():
        dict_ = {}
        dict_["timestamp"] = timestamp
        pd_dict_list.append({**dict_, **kvs})

    raw_df = pd.DataFrame(pd_dict_list)

    print("Before imputation:", raw_df.shape)
    print(raw_df.shape)
    # df.head()

    raw_df.to_csv(out_csv_file+".csv", index=False)

    # 1. Select partial columns
    raw_df = raw_df[selected_cols]

    # 2. Impute missing values (`ffill` i.e., forward fill)
    raw_df, raw_df_flags = impute_missing_values(raw_df)
    print("After imputation:", raw_df.shape)
    raw_df.to_csv(out_csv_file+".csv", index=False)


if __name__ == "__main__":
    for root_in_dir in [
        "../data/openxc/nyc",
        "../data/openxc/india",
        "../data/openxc/taiwan"
    ]:
        print(root_in_dir)
        for in_json_file in os.listdir(root_in_dir):
            if in_json_file.endswith(".json"):
                print(in_json_file)
                in_dir = os.path.join(root_in_dir, in_json_file.split(".")[0])
                os.makedirs(in_dir, exist_ok=True)
                openxc_json2csv(
                    in_json_file=os.path.join(root_in_dir, in_json_file),
                    out_csv_file=os.path.join(
                        in_dir, in_json_file.split(".")[0])
                )
            print()
