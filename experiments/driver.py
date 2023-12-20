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

from tqdm import tqdm
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
        # Car-hacking datasets: fill missing values with 0
        if 'car-hacking' in dataset_name and 'bits' in dataset_name:
            df = df.fillna(0)

            if model_name == 'tabddpm':  # Replace `str` target column with `int`
                df[current_config.target_column] = df[current_config.target_column].map(
                    current_config.target_column_mapping)

        print(df.shape)
        print(df.columns)

    start_time = time.time()
    end_time_train = None

    if model_name == "realtabformer-tabular":
        from realtabformer import REaLTabFormer
        from transformers.models.gpt2 import GPT2Config

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
            eval_steps=current_config.eval_steps,
            numeric_max_len=getattr(current_config, 'numeric_max_len', 10)
        )

        rtf_model.fit(df, num_bootstrap=current_config.num_bootstrap)
        rtf_model.save(os.path.join(work_folder, "rtf_model"))
        end_time_train = time.time()
        syn_df = rtf_model.sample(n_samples=len(df), gen_batch=1024)

    if model_name == "realtabformer-timeseries":
        from realtabformer import REaLTabFormer
        from pathlib import Path
        from transformers import EncoderDecoderConfig
        from transformers.models.gpt2 import GPT2Config

        join_on = "session_id"
        session_keys = current_config.session_keys
        timeseries_keys = list(set(df.columns) - set(session_keys + [join_on]))
        print("Raw df:", df.columns, df.shape, df.isna().sum().sum())

        # Metadata (parent table)
        parent_df = pd.DataFrame(df.groupby(list(
            session_keys+[join_on])).groups.keys(), columns=list(session_keys+[join_on]))
        print("parent_df:", parent_df.columns,
              parent_df.shape, df.isna().sum().sum())

        # Timeseries (child table)
        child_df = df[list(timeseries_keys+[join_on])]
        print("child_df:", child_df.columns,
              child_df.shape, df.isna().sum().sum())

        # Make sure that the key columns in both the
        # parent and the child table have the same name.
        assert ((join_on in parent_df.columns) and
                (join_on in child_df.columns))

        # Non-relational or parent table. Don't include the
        # unique_id field.
        parent_model = REaLTabFormer(
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
            batch_size=64,
            random_state=current_config.random_state,
            logging_steps=current_config.logging_steps,
            save_steps=current_config.save_steps,
            save_total_limit=current_config.save_total_limit,
            eval_steps=current_config.eval_steps,
            numeric_max_len=getattr(current_config, 'numeric_max_len', 10)
        )
        parent_model.fit(
            parent_df.drop(join_on, axis=1),
            num_bootstrap=current_config.num_bootstrap)

        pdir = Path(os.path.join(work_folder, "rtf_parent/"))
        parent_model.save(pdir)

        # # Get the most recently saved parent model,
        # # or a specify some other saved model.
        # parent_model_path = pdir / "idXXX"
        parent_model_path = sorted([
            p for p in pdir.glob("id*") if p.is_dir()],
            key=os.path.getmtime)[-1]

        child_model = REaLTabFormer(
            model_type="relational",
            relational_config=EncoderDecoderConfig(
                encoder=GPT2Config(
                    n_layer=getattr(current_config, 'n_layer', 12),
                    n_head=getattr(current_config, 'n_head', 12),
                    n_embd=getattr(current_config, 'n_embd', 768)
                ).to_dict(),
                decoder=GPT2Config(
                    n_layer=getattr(current_config, 'n_layer', 12),
                    n_head=getattr(current_config, 'n_head', 12),
                    n_embd=getattr(current_config, 'n_embd', 768)
                ).to_dict(),
            ),
            # set to `None` if experiencing the issue when fitting the child model: https://github.com/worldbank/REaLTabFormer/issues/22
            # parent_realtabformer_path=parent_model_path,
            parent_realtabformer_path=None,
            checkpoints_dir=os.path.join(work_folder, "rtf_checkpoints"),
            samples_save_dir=os.path.join(work_folder, "rtf_samples"),
            output_max_length=2048,
            train_size=0.8,
            gradient_accumulation_steps=4,
            epochs=current_config.epochs,
            batch_size=16,
            random_state=current_config.random_state,
            logging_steps=current_config.logging_steps,
            save_steps=current_config.save_steps,
            save_total_limit=current_config.save_total_limit,
            eval_steps=current_config.eval_steps,
            numeric_max_len=getattr(current_config, 'numeric_max_len', 10)
        )

        child_model.fit(
            df=child_df,
            in_df=parent_df,
            join_on=join_on,
            num_bootstrap=current_config.num_bootstrap)

        pdir = Path(os.path.join(work_folder, "rtf_child/"))
        child_model.save(pdir)

        # Generate parent samples.
        print("Start generating parent samples...")
        parent_samples = parent_model.sample(len(parent_df), gen_batch=512)
        print("Finish generating parent samples...")

        # Create the unique ids based on the index.
        parent_samples.index.name = join_on
        parent_samples = parent_samples.reset_index()

        # Generate the relational observations.
        print("Start generating child samples...")
        child_samples = child_model.sample(
            input_unique_ids=parent_samples[join_on],
            input_df=parent_samples.drop(join_on, axis=1),
            gen_batch=16)
        child_samples.index.name = join_on
        print("Finish generating child samples...")

        print("Parent df:", parent_samples.shape, parent_samples.columns)
        print(parent_samples.head())
        print("Child df:", child_samples.shape, child_samples.columns)
        print(child_samples.head())

        # merge parent and child tables
        syn_df = child_samples.merge(parent_samples, on=join_on)
        assert syn_df.shape[0] == child_samples.shape[0]  # sanity check

        # Child table is huge... cleaning up GPU memory for BERT inference later
        print("Initial GPU Usage")
        gpu_usage()
        del parent_model, child_model
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU Usage after emptying the cache")
        gpu_usage()

    if model_name == "ctgan":
        from ctgan import CTGAN

        discrete_columns = current_config['discrete_columns']
        print("discrete_columns:", discrete_columns)

        ctgan = CTGAN(epochs=100, verbose=True)

        print("Start CTGAN training...")
        ctgan.fit(df, discrete_columns)
        end_time_train = time.time()
        print("CTGAN training finished...")

        ctgan.save(os.path.join(work_folder, "model.pt"))
        print("CTGAN model saved...")

        syn_df = ctgan.sample(len(df))
        print("CTGAN sampling finished...")

    if model_name == "netshare":
        import netshare.ray as ray
        from netshare import Generator

        ray.config.enabled = False
        ray.init(address="auto")
        generator = Generator(config=current_config_file_path)
        generator.train_and_generate(work_folder=work_folder)
        syn_df = pd.read_csv(
            generator._pre_post_processor.best_syndf_filename_list[0])
        ray.shutdown()

    if model_name == "tabddpm":
        data_split = [0.7, 0.3, 0]  # train, val, test; no test in our case
        type_map = {'train': 0, 'val': 1, 'test': 2}

        # remove constant columns
        # df, constant_columns_mapping = remove_constant_columns(df)
        # for constant_col, val in constant_columns_mapping.items():
        #     if constant_col in current_config.discrete_columns:
        #         current_config.discrete_columns.remove(constant_col)

        def split_dataframe(input_df, ratios, seed=42):
            df = input_df.copy(deep=True)
            assert sum(ratios) == 1.0, "Sum of ratios must be 1.0"

            # Compute the split sizes
            train_ratio, val_ratio, test_ratio = ratios

            if test_ratio == 0:
                # Split the data into train and validation sets
                train, val = train_test_split(
                    df, test_size=val_ratio, random_state=seed)
                return list(train.index), list(val.index), []
            else:
                # First, split the data into train+val and test
                train_val, test = train_test_split(
                    df, test_size=test_ratio, random_state=seed)

                # Compute the new ratio for the validation set (original ratio relative to train+val)
                relative_val_ratio = val_ratio / (1 - test_ratio)

                # Split the train+val set into train and validation sets
                train, val = train_test_split(
                    train_val, test_size=relative_val_ratio, random_state=seed)

                return list(train.index), list(val.index), list(test.index)

        # train/val/test
        train_idx, val_idx, test_idx = split_dataframe(df, data_split)
        print("train_num, val_num, test_num: ", len(
            train_idx), len(val_idx), len(test_idx))

        os.makedirs(os.path.join(work_folder, "raw_dataset"),
                    exist_ok=True)  # preprocessed raw dataset

        # Categorical
        categorical_columns = copy.deepcopy(current_config.discrete_columns)
        if current_config.target_column in categorical_columns:
            categorical_columns.remove(current_config.target_column)
        print("Categorical columns: ", categorical_columns)
        categorical_array = df[categorical_columns].to_numpy(dtype=str)
        print("Catetgorical array shape:", categorical_array.shape)
        np.save(os.path.join(work_folder, "raw_dataset",
                "X_cat_train.npy"), categorical_array[train_idx])
        np.save(os.path.join(work_folder, "raw_dataset",
                "X_cat_val.npy"), categorical_array[val_idx])
        np.save(os.path.join(work_folder, "raw_dataset",
                "X_cat_test.npy"), categorical_array[test_idx])

        # Continuous
        numeric_columns = sorted(
            list(set(df.columns) - set(current_config.discrete_columns)))
        if current_config.target_column in numeric_columns:
            numeric_columns.remove(current_config.target_column)
        print("Numeric columns:", numeric_columns)
        numeric_array = df[numeric_columns].to_numpy()
        print("Numeric array shape: ", numeric_array.shape)
        np.save(os.path.join(work_folder, "raw_dataset",
                "X_num_train.npy"), numeric_array[train_idx])
        np.save(os.path.join(work_folder, "raw_dataset",
                "X_num_val.npy"), numeric_array[val_idx])
        np.save(os.path.join(work_folder, "raw_dataset",
                "X_num_test.npy"), numeric_array[test_idx])

        # y_col only used for regression task on val set (for picking the best model of generation)
        y_col = df[current_config.target_column]
        np.save(os.path.join(work_folder, "raw_dataset",
                "y_train.npy"), y_col[train_idx])
        np.save(os.path.join(work_folder, "raw_dataset",
                "y_val.npy"), y_col[val_idx])
        np.save(os.path.join(work_folder, "raw_dataset",
                "y_test.npy"), y_col[test_idx])

        # Config native to tabddpm
        config_info = {
            "task_type": "multiclass",
            "name": f"{model_name}_{dataset_name}",
            "id": f"{cur_time}",
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
            "n_num_features": len(numeric_columns),
            "n_cat_features": len(categorical_columns),
        }

        # Export info json
        with open(os.path.join(work_folder, "raw_dataset", "info.json"), 'w') as f:
            json.dump(config_info, f, indent=4)

        # Read basic toml
        basic_toml_config = toml.load('src/tabddpm/base-config.toml')
        os.makedirs(os.path.join(work_folder, "results"),
                    exist_ok=True)  # results folder
        toml_config = copy.deepcopy(basic_toml_config)
        toml_config['parent_dir'] = os.path.abspath(
            os.path.join(work_folder, "results"))
        toml_config['real_data_path'] = os.path.abspath(
            os.path.join(work_folder, "raw_dataset"))
        toml_config['num_numerical_features'] = len(numeric_columns)
        toml_config['device'] = "cuda:0"
        toml_config['sample']['num_samples'] = len(df)

        if 'openxc' in dataset_name:
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [256, 256]
            toml_config['model_params']['num_classes'] = current_config.num_classes
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif 'car-hacking' in dataset_name:
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = 5*[512]
            toml_config['model_params']['num_classes'] = current_config.num_classes
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001
        elif 'syncan' in dataset_name:
            toml_config['diffusion_params']['num_timesteps'] = 1000
            toml_config['model_params']['rtdl_params']['d_layers'] = [256, 256]
            toml_config['model_params']['num_classes'] = current_config.num_classes
            toml_config['train']['main']['batch_size'] = 1024
            toml_config['train']['main']['steps'] = 1000
            toml_config['train']['main']['lr'] = 0.001

        print(toml_config)
        with open(os.path.join(work_folder, 'config.toml'), 'w') as f:
            toml.dump(toml_config, f)

        # Run TabDDPM (use conda env `tabddpm`)
        exec_cmd(
            f"cd src/tabddpm && python3 scripts/pipeline.py \
                --config {os.path.abspath(os.path.join(work_folder, 'config.toml'))} \
                --sample \
                --train"
        )

        syn_numerical = np.load(os.path.join(
            toml_config['parent_dir'], 'X_num_train.npy'))
        print(syn_numerical[:2, :])
        print("Numerical columns", numeric_columns)
        syn_categorical = np.load(os.path.join(
            toml_config['parent_dir'], 'X_cat_train.npy'))
        print("Synthetic numerical array:", syn_numerical.shape)
        print("Synthetic categorical array:", syn_categorical.shape)
        syn_target = np.load(os.path.join(
            toml_config['parent_dir'], 'y_train.npy'))

        syn_df = pd.concat(
            [pd.DataFrame(syn_numerical, columns=numeric_columns),
             pd.DataFrame(syn_categorical,
                          columns=categorical_columns),
             pd.DataFrame(syn_target, columns=[current_config.target_column])],
            axis=1)
        print("Synthetic df:", syn_df.shape, syn_df.columns)
        print(syn_df.head())

    # ==========================================================================
    # =================Postprocess synthetic data===============================
    # ==========================================================================
    # Car-hacking datasets
    if 'car-hacking' in dataset_name and 'bits' in dataset_name:
        def postprocess_car_hacking(syn_df):
            # Function to convert CAN_ID fields to a hex string
            def can_id_bin2hex(row):
                binary_string = ''.join(
                    [str(int(row[f'CAN_ID_{i}'])) for i in range(11)])
                return '{:04x}'.format(int(binary_string, 2))

            # Function to convert DATA fields based on DLC
            def data_fields_bin2hex(row):
                bits_to_consider = int(row['DLC']) * 8
                data_bits = ''.join(
                    [str(int(row[f'DATA_{i}'])) for i in range(bits_to_consider)])
                return [f'{int(data_bits[i:i+8], 2):02x}' for i in range(0, len(data_bits), 8)]

            # Convert DLC field to int
            syn_df['DLC'] = syn_df['DLC'].astype(int)
            # Convert CAN_ID_i and DATA_i fields to int in case they are not
            for i in range(11):
                syn_df[f'CAN_ID_{i}'] = syn_df[f'CAN_ID_{i}'].astype(
                    float).astype(int)
            for i in range(64):
                syn_df[f'DATA_{i}'] = syn_df[f'DATA_{i}'].astype(
                    float).astype(int)

            converted_df = pd.DataFrame()
            converted_df['Timestamp'] = syn_df['Timestamp']
            converted_df['CAN_ID'] = syn_df.apply(can_id_bin2hex, axis=1)
            converted_df['DLC'] = syn_df['DLC']

            # Convert DATA fields
            data_transformed = syn_df.apply(
                data_fields_bin2hex, axis=1).tolist()
            for i in range(8):
                converted_df[f'DATA_{i}'] = [item[i] if i < len(
                    item) else '' for item in tqdm(data_transformed)]

            # Specify DATA_X columns as object types to avoid implicit conversion to float
            for i in range(8):
                converted_df[f'DATA_{i}'] = converted_df[f'DATA_{i}'].astype(
                    object)

            # Copy the Label column
            converted_df['Label'] = syn_df['Label']

            if model_name == 'tabddpm':
                # Map int target column back to str
                converted_df['Label'] = converted_df['Label'].map(
                    {v: k for k, v in current_config.target_column_mapping.items()})

            return converted_df

        syn_df = postprocess_car_hacking(syn_df)

    # SynCAN datasets with flags added
    elif 'syncan-flag' in dataset_name:
        for col in ['Signal1', 'Signal2', 'Signal3', 'Signal4']:
            syn_df.loc[syn_df[f'{col}_Missing'] == 1, col] = np.nan

        # Removing the SignalX_Missing columns
        syn_df.drop(columns=[f'{col}_Missing' for col in [
                    'Signal1', 'Signal2', 'Signal3', 'Signal4']], inplace=True)

        syn_df['ID'] = syn_df['ID'].astype(int)  # sanity conversion

    # Sessionized OpenXC datasets (NetShare, RTF-Time)
    elif 'openxc' in dataset_name and 'sessionized' in dataset_name:
        if 'accelerator_pedal_position' not in syn_df.columns:
            syn_df['accelerator_pedal_position_binned'] = syn_df['accelerator_pedal_position_binned'].astype(
                int)

            raw_df = pd.read_csv(
                current_config.global_config.original_data_file)
            _, bin_edges = pd.cut(raw_df['accelerator_pedal_position'],
                                  bins=current_config.pre_post_processor.config.sessionize.n_bins, retbins=True)
            # Function to sample a random value within the bin range

            def sample_from_bin(bin_index):
                if bin_index >= len(bin_edges) - 1:
                    return np.nan
                lower_bound = bin_edges[bin_index]
                upper_bound = bin_edges[bin_index + 1]
                return np.random.uniform(lower_bound, upper_bound)

            syn_df['accelerator_pedal_position'] = syn_df['accelerator_pedal_position_binned'].apply(
                sample_from_bin)
        else:
            print("accelerator_pedal_position already exists in the dataset")

    if args.order_csv_by_timestamp:
        # sort by timestamp
        syn_df = syn_df.sort_values(by=current_config.timestamp_colname)

    # Export synthetic csv to the target folder
    syn_df.to_csv(os.path.join(RESULT_PATH[args.config_partition]['csv'],
                  f'{model_name}_{dataset_name}_{cur_time}.csv'), index=False)
    print("Synthetic csv exported to:", os.path.join(
        RESULT_PATH[args.config_partition]['csv'], f'{model_name}_{dataset_name}_{cur_time}.csv'))

    # Export running time
    end_time = time.time()
    time_elapsed = end_time - start_time
    with open(os.path.join(RESULT_PATH[args.config_partition]['time'], f'{model_name}_{dataset_name}_{cur_time}.txt'), 'w') as f:
        f.write(f"{time_elapsed:.2f} seconds\n")
        f.write(f"{time_elapsed / 3600:.2f} hours\n")
        f.write(
            f"start_time: {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        if end_time_train is not None:
            f.write(
                f"end_time_train: {datetime.datetime.fromtimestamp(end_time_train).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"end_time: {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")


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
