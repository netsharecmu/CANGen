import json
import os
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm

from util import impute_missing_values


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

    # raw_df.to_csv(out_csv_file+".csv", index=False)

    # 1. Select partial columns
    raw_df = raw_df[selected_cols]
    raw_df.to_csv(out_csv_file+"_before_imputation.csv", index=False)

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
