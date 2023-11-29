import pandas as pd
import numpy as np
import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter


def hex_to_binary(hex_str, num_bits):
    return bin(int(hex_str, 16))[2:].zfill(num_bits)


def parse_row(row, can_id_bits=11):  # Set can_id_bits to 29 for extended CAN IDs
    timestamp = row['Timestamp']
    can_id_binary = hex_to_binary(row['CAN_ID'], can_id_bits)
    dlc = int(row['DLC'])
    # Assuming the data starts at the 4th column
    data_bytes = [row[f'DATA_{i}'] for i in range(dlc)]
    data_bits = ''.join([hex_to_binary(byte, 8) for byte in data_bytes])
    # Pad with NaN if less than 64 bits
    data_bits_padded = [int(bit) for bit in data_bits] + \
        [np.nan] * (64 - len(data_bits))

    if 'Label' in row:
        return [timestamp] + [int(bit) for bit in can_id_binary] + [dlc] + data_bits_padded + [row['Label']]
    else:
        return [timestamp] + [int(bit) for bit in can_id_binary] + [dlc] + data_bits_padded


def align_csv(input_csv, output_csv, can_id_bits=11):
    '''
        Align each row of the csv file to the same number of columns
        E.g.,
            1478191030.066350,0545,8,d8,59,00,84,00,00,00,00,R
            1478191030.067627,05f0,2,01,00,R
    '''
    # Parse the CSV file into a list of lists (not using pandas for variable length rows)
    with open(input_csv, 'r') as f:
        lines = [line.strip().split(',') for line in f]

    is_labelled = len(lines[0]) == 3 + int(lines[0][2]) + 1

    # Process each line
    data_rows = []
    for line in tqdm(lines):
        row = {}
        row['Timestamp'], row['CAN_ID'], row['DLC'] = line[0], line[1], line[2]
        for i in range(8):  # DATA
            if i < int(row['DLC']):
                row[f'DATA_{i}'] = line[3+i]
            else:
                row[f'DATA_{i}'] = np.nan
        if len(line) == 3 + int(row['DLC']) + 1:  # with Label
            row['Label'] = line[-1]
        elif len(line) == 3 + int(row['DLC']):  # without label
            pass
        else:
            raise ValueError("Length of line does not match!")

        data_rows.append(row)

    # Create DataFrame with binary bits
    column_names = ['Timestamp', 'CAN_ID', 'DLC'] + \
        [f'DATA_{i}' for i in range(8)] + \
        (['Label'] if is_labelled else [])
    final_df = pd.DataFrame(data_rows, columns=column_names)

    # Save the new DataFrame to CSV
    final_df.to_csv(output_csv, index=False)

    return final_df


def convert_csv(input_csv, output_csv, can_id_bits=11):
    df = pd.read_csv(input_csv)

    # Process each row
    data_rows = [parse_row(row) for idx, row in tqdm(df.iterrows())]

    # Create DataFrame with binary bits
    column_names = ['Timestamp'] + \
        [f'CAN_ID_{i}' for i in range(can_id_bits)] + \
        ['DLC'] + \
        [f'DATA_{i}' for i in range(64)] + \
        (['Label'] if 'Label' in df.columns else [])
    final_df = pd.DataFrame(data_rows, columns=column_names)

    # Save the new DataFrame to CSV
    final_df.to_csv(output_csv, index=False)

    return final_df


if __name__ == "__main__":
    for csv_name in ['DoS', 'Fuzzy', 'RPM', 'gear']:
        print(csv_name)

        # 1. Align the CSV file to have the same number of columns
        print("Aligning...")
        align_csv(
            input_csv=f"../data/car_hacking/{csv_name}_dataset.csv",
            output_csv=f"../data/car_hacking/{csv_name}_dataset_aligned.csv",
        )

        # 2. Sample and create train/test split
        print("Sampling...")
        df = pd.read_csv(
            f"../data/car_hacking/{csv_name}_dataset_aligned.csv")
        df = df.sample(n=1000000, random_state=42).reset_index(drop=True)

        train_set, test_set = train_test_split(
            df, test_size=0.3, random_state=42)
        train_set = train_set.sort_values('Timestamp')
        test_set = test_set.sort_values('Timestamp')
        print(train_set.shape, Counter(train_set['Label']))
        print(test_set.shape, Counter(test_set['Label']))

        train_set.to_csv(
            f"../data_selected/car_hacking/{csv_name}_dataset_aligned_train.csv", index=False)
        test_set.to_csv(
            f"../data_selected/car_hacking/{csv_name}_dataset_aligned_test.csv", index=False)

        # 3. Convert to binary
        print("Converting to binary...")
        convert_csv(
            input_csv=f"../data_selected/car_hacking/{csv_name}_dataset_aligned_train.csv",
            output_csv=f"../data_selected/car_hacking/{csv_name}_dataset_aligned_train_bits.csv",
        )
        convert_csv(
            input_csv=f"../data_selected/car_hacking/{csv_name}_dataset_aligned_test.csv",
            output_csv=f"../data_selected/car_hacking/{csv_name}_dataset_aligned_test_bits.csv",
        )

        print()
