import os
import pandas as pd
import numpy as np

from tqdm import tqdm


def hex_to_binary(hex_str, num_bits):
    return bin(int(hex_str, 16))[2:].zfill(num_bits)


def parse_row(row, has_subclass):
    # Convert Arbitration_ID to 11-bit binary
    can_id_binary = hex_to_binary(row["Arbitration_ID"], 11)

    # Convert Data to 64-bit binary, pad with NaN for missing bits
    if row['DLC'] == 0:
        data_bits_padded = [np.nan] * 64
    else:
        data_bytes = row['Data'].split(' ')
        data_bits = ''.join([hex_to_binary(byte, 8) for byte in data_bytes])
        data_bits_padded = [int(bit) for bit in data_bits] + \
            [np.nan] * (64 - len(data_bits))

    # Construct the parsed row
    parsed_row = [row['Timestamp']] + [int(bit) for bit in can_id_binary] + [
        row['DLC']] + data_bits_padded + [row['Class']]
    if has_subclass:
        parsed_row.append(row.get('SubClass', np.nan))

    return parsed_row


def convert_to_csv(input_file, output_file=None):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Determine if 'SubClass' column exists
    has_subclass = 'SubClass' in df.columns

    # Parse each row
    parsed_data = [parse_row(row, has_subclass)
                   for index, row in tqdm(df.iterrows())]

    # Define column names
    column_names = ['Timestamp'] + [f'CAN_ID_{i}' for i in range(
        11)] + ['DLC'] + [f'DATA_{i}' for i in range(64)] + ['Class']
    if has_subclass:
        column_names.append('SubClass')

    # Create new DataFrame
    new_df = pd.DataFrame(parsed_data, columns=column_names)

    # Save DataFrame to CSV
    if output_file is not None:
        new_df.to_csv(output_file, index=False)

    return new_df

# Example usage
# convert_to_csv('input_file.csv', 'output_file.csv')


if __name__ == "__main__":
    for csv_file in os.listdir("../data/car_challenge/raw/0_Preliminary/0_Training"):
        if csv_file.endswith(".csv"):
            print(csv_file)
            convert_to_csv(
                os.path.join(
                    "../data/car_challenge/raw/0_Preliminary/0_Training", csv_file),
                os.path.join("../data/car_challenge", csv_file)
            )
            print()

    for csv_file in os.listdir("../data/car_challenge/raw/0_Preliminary/1_Submission"):
        if csv_file.endswith(".csv"):
            print(csv_file)
            convert_to_csv(
                os.path.join(
                    "../data/car_challenge/raw/0_Preliminary/1_Submission", csv_file),
                os.path.join("../data/car_challenge", csv_file)
            )
            print()

    for csv_file in os.listdir("../data/car_challenge/raw/1_Final"):
        if csv_file.endswith(".csv"):
            print(csv_file)
            convert_to_csv(
                os.path.join("../data/car_challenge/raw/1_Final", csv_file),
                os.path.join("../data/car_challenge", csv_file)
            )
            print()
