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
import random
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


def main(args):
    # Global timestamp
    if args.cur_time is None:
        cur_time = current_timestamp()  # use provided current timestamp from the job script
    else:
        # if not provided, just use the current timestamp when running this script
        cur_time = args.cur_time

    # Load configs
    if args.config_partition == 'small-scale':
        from config_small_scale import configs
        from config_small_scale import CANGen_BASE_FOLDER
    else:
        raise ValueError(f"Unknown config partition: {args.config_partition}")

    RESULT_PATH_BASE = os.path.join(
        CANGen_BASE_FOLDER, "results", "vehiclesec2024")
    RESULT_PATH_BASE_SMALL_SCALE = os.path.join(
        RESULT_PATH_BASE, "small-scale")
    RESULT_PATH = {
        'small-scale': {
            # models/logs/intermediate files etc.
            'runs': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, "runs"),
            # raw/synthetic csv files
            'csv': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, 'csv'),
            # time elapsed for each model
            'time': os.path.join(RESULT_PATH_BASE_SMALL_SCALE, 'time'),
        }
    }

    for section, section_path in RESULT_PATH.items():
        for sub_section, sub_section_path in section_path.items():
            os.makedirs(sub_section_path, exist_ok=True)

    model_name, dataset_name = args.model_name, args.dataset_name

    work_folder = os.path.join(
        RESULT_PATH[args.config_partition]['runs'],
        f'{model_name}_{dataset_name}_{cur_time}')
    os.makedirs(work_folder, exist_ok=True)

    # save configs to work folder
    current_config = Config(configs[model_name][dataset_name])
    current_config.dump_to_file(os.path.join(
        work_folder, "config_driver.json"))
    current_config_file_path = os.path.join(work_folder, "config_driver.json")

    # ==========================================================================
    # ================================Run models================================
    # ==========================================================================
    # READ RAW DATA FILE and select related columns (e.g., drop `version`/`ihl`/`chksum`)
    if 'raw_csv_file' in current_config:  # realtabformer-tabular, realtabformer-timeseries, ctgan, tvae, tabddpm, crossformer, d3vae, scinet, dlinear, patchtst
        df = pd.read_csv(current_config.raw_csv_file)
        print(df.shape)
        print(df.columns)

    start_time = time.time()
    end_time_train = None

    if model_name == "realtabformer-tabular":
        from realtabformer import REaLTabFormer
        from transformers.models.gpt2 import GPT2Config

        print("Random state:", current_config.random_state)

        # Non-relational or parent table.
        rtf_model = REaLTabFormer(
            model_type="tabular",
            tabular_config=GPT2Config(
                n_layer=getattr(current_config, 'n_layer', 12),
                n_head=getattr(current_config, 'n_head', 12),
                n_embd=getattr(current_config, 'n_embd', 768)
            ),
            checkpoints_dir=os.path.join(work_folder, "rtf_checkpoints"),
            samples_save_dir=os.path.join(work_folder, "rtf_samples"),
            gradient_accumulation_steps=4,
            epochs=current_config.epochs,
            batch_size=16,
            random_state=current_config.random_state,
            logging_steps=current_config.logging_steps,
            save_steps=current_config.save_steps,
            save_total_limit=current_config.save_total_limit,
            eval_steps=current_config.eval_steps)

        rtf_model.fit(df, num_bootstrap=current_config.num_bootstrap)
        rtf_model.save(os.path.join(work_folder, "rtf_model"))
        syn_df = rtf_model.sample(n_samples=len(df), gen_batch=1024)

    # ==========================================================================
    # =================Postprocess synthetic data===============================
    # ==========================================================================
    # Export synthetic csv to the target folder
    syn_df.to_csv(os.path.join(RESULT_PATH[args.config_partition]['csv'],
                  f'{model_name}_{dataset_name}_{cur_time}.csv'), index=False)
    print("Synthetic csv exported to:", os.path.join(
        RESULT_PATH[args.config_partition]['csv'], f'{model_name}_{dataset_name}_{cur_time}.csv'))


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

    args = parser.parse_args()
    main(args)
