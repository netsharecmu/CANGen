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

    # OpenXC (Sessionized)
    'openxc-nyc-downtown-east-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'nyc_downtown_east_sessionized.csv'
    ),
    'openxc-india-new-delhi-railway-to-aiims-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'india_New_Delhi_Railway_to_AIIMS_sessionized.csv'
    ),
    'openxc-taiwan-highwayno2-can-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'taiwan_HighwayNo2_can_sessionized.csv'
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
    'car-hacking-dos-bits-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'DoS_dataset_aligned_train_bits_sessionized.csv'
    ),
    'car-hacking-fuzzy-bits-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'Fuzzy_dataset_aligned_train_bits_sessionized.csv'
    ),
    'car-hacking-rpm-bits-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'RPM_dataset_aligned_train_bits_sessionized.csv'
    ),
    'car-hacking-gear-bits-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'car_hacking', 'gear_dataset_aligned_train_bits_sessionized.csv'
    ),

    # SynCAN
    'syncan-raw': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'syncan', 'train.csv'
    ),

    'syncan-flag': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'syncan', 'train_flags.csv'
    ),

    'syncan-flag-sessionized': os.path.join(
        CANGen_BASE_FOLDER, 'data_selected', 'syncan', 'train_flags_sessionized.csv'
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
            "n_layer": 3,
            "n_head": 4,
            "n_embd": 128,
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

# REaLTabFormer (relational)
configs['realtabformer-timeseries'] = Config()
for dataset_name, filename in DICT_DATASET_FILENAME.items():
    if 'openxc' in dataset_name:
        session_keys = ['brake_pedal_status',
                        'accelerator_pedal_position_binned']
    elif 'car-hacking' in dataset_name:
        session_keys = [f"CAN_ID_{i}" for i in range(11)]
    elif 'syncan' in dataset_name:
        session_keys = ['ID']

    configs['realtabformer-timeseries'][dataset_name] = Config(
        {
            "raw_csv_file": filename,
            "session_keys": session_keys,
            "n_layer": 3,
            "n_head": 4,
            "n_embd": 128,
            "logging_steps": 1000,
            "save_steps": 10000,
            "save_total_limit": 10,
            "save_total_limit": None,
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
for dataset_name in ['syncan-flag']:
    configs['ctgan'][dataset_name] = Config(
        {
            "raw_csv_file": DICT_DATASET_FILENAME[dataset_name],
            "discrete_columns": ['Label', 'ID'] +
            [f'Signal{i+1}_Missing' for i in range(4)],
            "timestamp_colname": get_timestamp_colname(dataset_name)
        }
    )

# NetShare
configs['netshare'] = Config()
base_config = Config.load_from_file(
    'src/netshare/netshare/configs/default/single_event_per_row.json'
)
# OpenXC
for dataset_name in [
    'openxc-nyc-downtown-east-sessionized',
    'openxc-india-new-delhi-railway-to-aiims-sessionized',
    'openxc-taiwan-highwayno2-can-sessionized'
]:
    c = copy.deepcopy(base_config)
    c.global_config.original_data_file = DICT_DATASET_FILENAME[dataset_name]
    c.global_config.overwrite = True
    c.pre_post_processor.config.truncate = 'none'
    c.global_config.n_chunks = 1
    c.model.config.batch_size = 512
    c.model.config.sample_len = [1]

    c.model.config.epochs = 1000
    c.model.config.iterations = None
    c.model.config.epoch_checkpoint_freq = 50
    c.model.config.max_train_time = None

    c.pre_post_processor.config.sessionize = {
        "n_bins": 20,
        "num_per_session": 10
    }

    c.pre_post_processor.config.timestamp = {
        "column": get_timestamp_colname(dataset_name),
        "generation": True,
        "encoding": "interarrival",
        "normalization": "ZERO_ONE"
    }
    c.pre_post_processor.config.metadata = [
        {
            "column": "brake_pedal_status",
            "type": "integer",
            "encoding": "categorical"
        },
        {
            "column": "accelerator_pedal_position_binned",
            "type": "integer",
            "encoding": "categorical"
        },
        {
            "column": "session_id",
            "type": "float",
            "normalization": "ZERO_ONE",
        }
    ]
    c.pre_post_processor.config.timeseries = [
        {
            "column": "vehicle_speed",
            "type": "float",
            "normalization": "ZERO_ONE",
        },
        {
            "column": "engine_speed",
            "type": "float",
            "normalization": "ZERO_ONE",
        },
        {
            "column": "accelerator_pedal_position",
            "type": "float",
            "normalization": "ZERO_ONE",
        },
        {
            "column": "transmission_gear_position",
            "type": "string",
            "encoding": "categorical"
        },
    ]
    c.timestamp_colname = get_timestamp_colname(dataset_name)

    configs['netshare'][dataset_name] = c

# Car-hacking
for dataset_name in [
    'car-hacking-dos-bits-sessionized',
    'car-hacking-fuzzy-bits-sessionized',
    'car-hacking-rpm-bits-sessionized',
    'car-hacking-gear-bits-sessionized'
]:
    c = copy.deepcopy(base_config)
    c.global_config.original_data_file = DICT_DATASET_FILENAME[dataset_name]
    c.global_config.overwrite = True
    c.pre_post_processor.config.truncate = 'none'
    c.global_config.n_chunks = 1
    c.model.config.batch_size = 512
    c.model.config.sample_len = [1]

    c.model.config.epochs = 1000
    c.model.config.iterations = None
    c.model.config.epoch_checkpoint_freq = 50
    c.model.config.max_train_time = None

    c.pre_post_processor.config.sessionize = {
        "num_per_session": 10
    }

    c.pre_post_processor.config.timestamp = {
        "column": get_timestamp_colname(dataset_name),
        "generation": True,
        "encoding": "interarrival",
        "normalization": "ZERO_ONE"
    }
    c.pre_post_processor.config.metadata = [
        {
            "column": f"CAN_ID_{i}",
            "type": "integer",
            "encoding": "categorical"
        }
        for i in range(11)
    ] + \
        [{
            "column": "session_id",
            "type": "float",
            "normalization": "ZERO_ONE",
        }]
    c.pre_post_processor.config.timeseries = [
        {
            "column": "DLC",
            "type": "integer",
            "encoding": "categorical"
        },
        {
            "column": "Label",
            "type": "string",
            "encoding": "categorical"
        },
    ] + \
        [
        {
            "column": f"DATA_{i}",
            "type": "integer",
            "encoding": "categorical"
        }
        for i in range(64)
    ]
    c.timestamp_colname = get_timestamp_colname(dataset_name)

    configs['netshare'][dataset_name] = c

for dataset_name in [
    'syncan-flag-sessionized'
]:
    c = copy.deepcopy(base_config)
    c.global_config.original_data_file = DICT_DATASET_FILENAME[dataset_name]
    c.global_config.overwrite = True
    c.pre_post_processor.config.truncate = 'none'
    c.global_config.n_chunks = 1
    c.model.config.batch_size = 512
    c.model.config.sample_len = [1]

    c.model.config.epochs = 1000
    c.model.config.iterations = None
    c.model.config.epoch_checkpoint_freq = 50
    c.model.config.max_train_time = None

    c.pre_post_processor.config.sessionize = {
        "num_per_session": 10
    }

    c.pre_post_processor.config.timestamp = {
        "column": get_timestamp_colname(dataset_name),
        "generation": True,
        "encoding": "interarrival",
        "normalization": "ZERO_ONE"
    }
    c.pre_post_processor.config.metadata = [
        {
            "column": "ID",
            "type": "integer",
            "encoding": "categorical"
        },
        {
            "column": "session_id",
            "type": "float",
            "normalization": "ZERO_ONE",
        }
    ]
    c.pre_post_processor.config.timeseries = [
        {
            "column": f"Signal{i+1}",
            "type": "float",
            "normalization": "ZERO_ONE",
        }
        for i in range(4)
    ] + \
        [
        {
            "column": f"Signal{i+1}_Missing",
            "type": "integer",
            "encoding": "categorical"
        }
        for i in range(4)
    ] + \
        [
        {
            "column": "Label",
            "type": "string",
            "encoding": "categorical"
        }
    ]

    c.timestamp_colname = get_timestamp_colname(dataset_name)

    configs['netshare'][dataset_name] = c

# TabDDPM
configs['tabddpm'] = Config()
for dataset_name in [
    'openxc-nyc-downtown-east',
    'openxc-india-new-delhi-railway-to-aiims',
    'openxc-taiwan-highwayno2-can'
]:
    configs['tabddpm'][dataset_name] = Config(
        {
            "raw_csv_file": DICT_DATASET_FILENAME[dataset_name],
            "discrete_columns": ['brake_pedal_status', 'transmission_gear_position'],
            "target_column": 'brake_pedal_status',
            "num_classes": len(set(pd.read_csv(DICT_DATASET_FILENAME[dataset_name])['brake_pedal_status'])),
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
    configs['tabddpm'][dataset_name] = Config(
        {
            "raw_csv_file": DICT_DATASET_FILENAME[dataset_name],
            "discrete_columns": discrete_columns,
            "target_column": 'Label',
            "target_column_mapping": {
                'R': 0,  # normal
                'T': 1  # injected
            },
            "num_classes": len(set(pd.read_csv(DICT_DATASET_FILENAME[dataset_name])['Label'])),
            "timestamp_colname": get_timestamp_colname(dataset_name)
        }
    )
for dataset_name in ['syncan-flag']:
    configs['tabddpm'][dataset_name] = Config(
        {
            "raw_csv_file": DICT_DATASET_FILENAME[dataset_name],
            "discrete_columns": ['Label', 'ID'] +
            [f'Signal{i+1}_Missing' for i in range(4)],
            "target_column": 'Label',
            "num_classes": len(set(pd.read_csv(DICT_DATASET_FILENAME[dataset_name])['Label'])),
            "timestamp_colname": get_timestamp_colname(dataset_name)
        }
    )

configs.dump_to_file("small-scale-config.json")
