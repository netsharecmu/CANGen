'''
A hacky experiment configuration file. Probably a bad idea.
'''

import os
import copy
import toml
import json
import shutil

import pandas as pd

from config_io import Config

# Change this if you are working on different platforms
CANGen_BASE_FOLDER = '/ocean/projects/cis230033p/yyin4/CANGen'

# {model_name: [a list of configs]}. The first key is always the model name, for ease of parsing
configs = Config()

# DATASETS: Short descriptive name : file location
# Input this manually for datasets to be run
DATASET_NAME_FILE = {
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
        CANGen_BASE_FOLDER, 'data', 'openxc', 'taiwan', 'HighwayNo2_can', 'HighwayNo2_can_before_imputation.csv'
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
        CANGen_BASE_FOLDER, 'data-selected', 'syncan', 'train.csv'
    )

}
