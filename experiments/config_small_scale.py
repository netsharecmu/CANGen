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
# Input this manually
# OpenXC
DATASET_NAME_FILE = {}
DATASET_NAME_FILE['openxc-nyc-downtown-east'] = os.path.join(
    CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'nyc_downtown_east.csv'
)
DATASET_NAME_FILE['india-new-delhi-railway-to-aiims'] = os.path.join(
    CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'india_New_Delhi_Railway_to_AIIMS.csv'
)
DATASET_NAME_FILE['taiwan-highwayno2-can'] = os.path.join(
    CANGen_BASE_FOLDER, 'data_selected', 'openxc', 'taiwan_HighwayNo2_can.csv'
)
