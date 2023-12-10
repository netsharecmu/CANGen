'''
A hacky experiment configuration file. Probably a bad idea.
'''

import os
import copy
import toml
import json
import shutil
import random

import pandas as pd

from config_io import Config

# Change this if you are working on different platforms
CANGen_BASE_FOLDER = '/ocean/projects/cis230033p/yyin4/CANGen'

# {model_name: [a list of configs]}. The first key is always the model name, for ease of parsing
configs = Config()

# DATASETS: Short descriptive name : file location
# Input this manually for datasets to be run
DICT_DATASET_FILENAME = {
    # OpenXC
    'openxc-nyc-downtown-east': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'nyc_downtown_east.csv'
    ),
    'openxc-india-new-delhi-railway-to-aiims': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'india_New_Delhi_Railway_to_AIIMS.csv'
    ),
    'openxc-taiwan-highwayno2-can': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'taiwan_HighwayNo2_can.csv'
    ),
    'openxc-nyc-downtown-east-no-imputation': os.path.join(
        CANGen_BASE_FOLDER, 'data', 'openxc', 'nyc', 'downtown-east', 'downtown-east_before_imputation.csv'
    ),
    'openxc-india-new-delhi-railway-to-aiims-no-imputation': os.path.join(
        CANGen_BASE_FOLDER, 'data', 'openxc', 'india', 'New_Delhi_Railway_to_AIIMS', 'New_Delhi_Railway_to_AIIMS_before_imputation.csv'
    ),
    'openxc-taiwan-highwayno2-can-no-imputation': os.path.join(
        CANGen_BASE_FOLDER, 'data', 'openxc', 'taiwan', 'HighwayNo2-can', 'HighwayNo2-can_before_imputation.csv'
    ),

    # Car-hacking
    'car-hacking-dos-bits': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'DoS_dataset_aligned_train_bits.csv',
    ),
    'car-hacking-fuzzy-bits': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'Fuzzy_dataset_aligned_train_bits.csv',
    ),
    'car-hacking-rpm-bits': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'RPM_dataset_aligned_train_bits.csv',
    ),
    'car-hacking-gear-bits': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'gear_dataset_aligned_train_bits.csv',
    ),
    'car-hacking-dos-hex': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'DoS_dataset_aligned_train.csv',
    ),
    'car-hacking-fuzzy-hex': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'Fuzzy_dataset_aligned_train.csv',
    ),
    'car-hacking-rpm-hex': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'RPM_dataset_aligned_train.csv',
    ),
    'car-hacking-gear-hex': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'gear_dataset_aligned_train.csv',
    ),

    # SynCAN
    'syncan-raw': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'syncan', 'train.csv'
    )

}


def get_timestamp_colname(dataset_name):
    if 'openxc' in dataset_name:
        return 'timestamp'
    elif 'car-hacking' in dataset_name:
        return 'Timestamp'
    elif 'syncan' in dataset_name:
        return 'Time'
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


# RTF-TAB
configs['realtabformer-tabular'] = Config()
for dataset_name, filename in DICT_DATASET_FILENAME.items():
    configs['realtabformer-tabular'][dataset_name] = Config(
        {
            "raw_csv_file": filename,
            # "n_layer": 3,
            # "n_head": 4,
            # "n_embd": 128,
            "logging_steps": 1000,
            "save_steps": 10000,
            "save_total_limit": 10,
            "eval_steps": None,
            "epochs": 100,
            "num_bootstrap": 100,
            "random_state": random.randint(0, 2**16),
            "numeric_max_len": 15,
            "timestamp_colname": get_timestamp_colname(dataset_name)
        }
    )

# CTGAN
configs['ctgan'] = Config()
for dataset_name in [
    'openxc-nyc-downtown-east',
    'openxc-india-new-delhi-railway-to-aiims',
    'openxc-taiwan-highwayno2-can'
]:
    configs['ctgan'][dataset_name] = Config(
        {
            "raw_csv_file": DICT_DATASET_FILENAME[dataset_name],
            "discrete_columns": ['brake_pedal_status', 'transmission_gear_position'],
            "timestamp_colname": get_timestamp_colname(dataset_name)
        }
    )
for dataset_name in [
    'car-hacking-dos-bits',
    'car-hacking-fuzzy-bits',
    'car-hacking-rpm-bits',
    'car-hacking-gear-bits'
]:
    discrete_columns = \
        [f'CAN_ID_{i}' for i in range(11)] + \
        [f'DATA_{i}' for i in range(64)] + \
        ['Label']
    configs['ctgan'][dataset_name] = Config(
        {
            "raw_csv_file": DICT_DATASET_FILENAME[dataset_name],
            "discrete_columns": discrete_columns,
            "timestamp_colname": get_timestamp_colname(dataset_name)
        }
    )
