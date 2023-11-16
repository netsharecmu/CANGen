import pandas as pd
import numpy as np
import os

from tqdm import tqdm


def hex_to_binary(hex_str, num_bits):
    return bin(int(hex_str, 16))[2:].zfill(num_bits)


def parse_row(row, can_id_bits=11):  # Set can_id_bits to 29 for extended CAN IDs
    timestamp = row['Timestamp']
    can_id_binary = hex_to_binary(row['CAN_ID'], can_id_bits)
    dlc = row['DLC']
    data_bytes = row[3:3+dlc]  # Assuming the data starts at the 4th column
    data_bits = ''.join([hex_to_binary(byte, 8) for byte in data_bytes])
    # Pad with NaN if less than 64 bits
    data_bits_padded = [int(bit) for bit in data_bits] + \
        [np.nan] * (64 - len(data_bits))
    label = row['Label']
    return [timestamp] + [int(bit) for bit in can_id_binary] + data_bits_padded + [label]


def convert_csv(input_csv, output_csv, can_id_bits=11):
    # Read the CSV data into a pandas DataFrame
    df = pd.read_csv(input_csv, header=None, names=[
                     'Timestamp', 'CAN_ID', 'DLC', 'DATA_0', 'DATA_1', 'DATA_2', 'DATA_3', 'DATA_4', 'DATA_5', 'DATA_6', 'DATA_7', 'Label'])

    # Process each row
    data_rows = [parse_row(row, can_id_bits)
                 for index, row in tqdm(df.iterrows())]

    # Create DataFrame with binary bits
    column_names = ['Timestamp'] + [f'CAN_ID_{i}' for i in range(can_id_bits)] + [
        f'DATA_{i}' for i in range(64)]
    final_df = pd.DataFrame(data_rows, columns=column_names)

    # Save the new DataFrame to CSV
    final_df.to_csv(output_csv, index=False)

    return final_df


if __name__ == "__main__":
    for csv_name in ['DoS', 'Fuzzy', 'RPM', 'gear']:
        print(csv_name)
        convert_csv(
            input_csv=f"../data/car_hacking/{csv_name}_dataset.csv",
            output_csv=f"../data/car_hacking/{csv_name}_dataset_bits.csv",
        )
        print()
