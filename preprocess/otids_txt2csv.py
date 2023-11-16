import os
import csv

import pandas as pd
import numpy as np

from tqdm import tqdm


def hex_to_binary(hex_str, num_bits):
    return bin(int(hex_str, 16))[2:].zfill(num_bits)


def parse_line(line, can_id_bits=11):  # Set can_id_bits to 29 for extended CAN IDs
    parts = line.split()
    timestamp = float(parts[1])
    can_id_binary = hex_to_binary(parts[3], can_id_bits)
    dlc = int(parts[6])  # DLC is the 7th part
    data_bits = ''.join([hex_to_binary(byte, 8) for byte in parts[7:7+dlc]])
    # Pad with NaN if less than 64 bits
    data_bits_padded = [int(bit) for bit in data_bits] + \
        [np.nan] * (64 - len(data_bits))
    return [timestamp] + [int(bit) for bit in can_id_binary] + [dlc] + data_bits_padded


def convert_to_csv(input_file, output_file, can_id_bits=11):
    data_rows = []

    with open(input_file, 'r') as infile:
        for line in tqdm(infile):
            if line.strip():  # Ignore empty lines
                row = parse_line(line, can_id_bits)
                if row:  # Ensure row is not None
                    data_rows.append(row)

    # Create DataFrame
    column_names = ['Timestamp'] + [f'CAN_ID_{i}' for i in range(can_id_bits)] + \
                   ['DLC'] + [f'DATA_{i}' for i in range(64)]
    df = pd.DataFrame(data_rows, columns=column_names)

    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)

    # Check if the number of rows match
    assert len(data_rows) == pd.read_csv(output_file).shape[0], \
        "Number of rows in the CSV does not match the number of rows in the text file."


# Example usage
# convert_to_csv(
#     '/Users/sniperyyc/Documents/Research/CANGen/data/OTIDS/toy_input.txt',
#     '/Users/sniperyyc/Documents/Research/CANGen/data/OTIDS/output_data.csv')


if __name__ == "__main__":
    for txt_file in os.listdir("../data/otids"):
        if txt_file.endswith(".txt"):
            print(txt_file)
            convert_to_csv(
                os.path.join("../data/otids", txt_file),
                os.path.join("../data/otids", txt_file.split(".")[0]+".csv")
            )
            print()
