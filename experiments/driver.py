"""
Driver file for calling different generative models

Example usage:

python3 driver.py \
    --config_partition small-scale \
    --dataset_name ugr16 \
    --model_name netshare \
    --cur_time 0
"""

import os
import gc
import json
import time
import copy
import toml
import torch
import joblib
import datetime
import argparse
import subprocess

import pandas as pd
import numpy as np

from config_io import Config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from GPUtil import showUtilization as gpu_usage
# from config import configs # run once to ensure the latest configs are loaded
# print("Finish loading configs")
# from config_small_scale import NETGPT_BASE_FOLDER


def exec_cmd(cmd, wait=True):
    p = subprocess.Popen(cmd, shell=True)
    if wait:
        p.wait()
    returnCode = p.poll()
    if returnCode != 0:
        raise Exception(f"Command {cmd} failed with error code {returnCode}")
    return returnCode


def current_timestamp():
    now = datetime.datetime.now()
    # milliseconds precision
    return now.strftime("%Y%m%d%H%M%S") + str(now.microsecond // 1000).zfill(3)


def remove_constant_columns(df):
    constant_columns_mapping = {}
    for col in df.columns:
        if len(set(df[col])) == 1:
            constant_columns_mapping[col] = list(set(df[col]))[0]
    df = df.drop(columns=list(constant_columns_mapping.keys()))

    return df, constant_columns_mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script description')
    parser.add_argument('--config_partition', type=str,
                        help="Partiton of the config (e.g., small-scale, large-scale)")
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--model_name', type=str, help='Model name')
    # specify a special time for debugging/testing
    parser.add_argument('--cur_time', type=int, default=None,
                        help='Unix timestamp representing current time')
    parser.add_argument('--order_csv_by_timestamp', action='store_true',
                        help='Whether to order the synthetic csv by timestamp')
    parser.add_argument('--export_bert_embeddings', action='store_true',
                        help='Whether to export BERT embeddings')

    args = parser.parse_args()
    main(args)
