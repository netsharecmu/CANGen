import os

import pandas as pd
import numpy as np

from tqdm import tqdm


def hex_to_binary(hex_str, num_bits):
    return bin(int(hex_str, 16))[2:].zfill(num_bits)


def parse_line(line):
    parts = line.split()
    timestamp = float(parts[1])
    can_id = int(parts[3], 16)  # CAN ID is the 4th part
    dlc = int(parts[6])  # DLC is the 7th part
    data_bits = ''.join([hex_to_binary(byte, 8) for byte in parts[7:7+dlc]])
    # Pad with NaN if less than 64 bits
    data_bits_padded = [int(bit) for bit in data_bits] + \
        [np.nan] * (64 - len(data_bits))
    return [timestamp, can_id, dlc] + data_bits_padded


def convert_to_csv(input_file, output_file):
    data_rows = []

    with open(input_file, 'r') as infile:
        for line in tqdm(infile):
            if line.strip():  # Ignore empty lines
                row = parse_line(line)
                data_rows.append(row)

    # Create DataFrame
    column_names = ['Timestamp', 'CAN_ID', 'DLC'] + \
        [f'DATA_{i}' for i in range(64)]
    df = pd.DataFrame(data_rows, columns=column_names)

    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)

    return df, len(data_rows)


# # Example usage
# df, num_rows = convert_to_csv(
#     '/Users/sniperyyc/Documents/Research/CANGen/data/OTIDS/toy_input.txt', 'output_data.csv')

# # Print number of rows
# print(f"Number of rows in the original text file: {num_rows}")
# print(f"Number of rows in the final CSV file: {len(df)}")


if __name__ == "__main__":
    for txt_file in os.listdir("../data/OTIDS"):
        if txt_file.endswith(".txt"):
            print(txt_file)
            df, num_rows = convert_to_csv(
                os.path.join("../data/OTIDS", txt_file),
                os.path.join("../data/OTIDS", txt_file.split(".")[0]+".csv")
            )
            print(f"Number of rows in the original text file: {num_rows}")
            print(f"Number of rows in the final CSV file: {len(df)}")
            print()
