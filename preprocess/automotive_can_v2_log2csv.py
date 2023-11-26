import os
import pandas as pd

from tqdm import tqdm


def hex_to_binary(hex_str, num_bits):
    return bin(int(hex_str, 16))[2:].zfill(num_bits)


def parse_raw_line(line, can_id_bits):
    parts = line.strip().split(' ')
    # Remove parentheses and convert to float
    timestamp = float(parts[0][1:-1])
    can_id_hex = parts[2].split('#')[0]  # Extract CAN ID
    data_hex = parts[2].split('#')[1]
    dlc = len(data_hex) // 2  # Each byte is represented by two hex characters
    can_id_binary = hex_to_binary(can_id_hex, can_id_bits)
    data_binary = hex_to_binary(data_hex, dlc * 8)
    return [timestamp, can_id_binary, dlc, data_binary]


def convert_to_csv(input_file, output_file, can_id_bits=11):
    data_rows = []

    with open(input_file, 'r') as infile:
        for line in tqdm(infile):
            if line.strip():  # Ignore empty lines
                parsed_row = parse_raw_line(line, can_id_bits)
                data_rows.append(parsed_row)

    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=[
                      'Timestamp', 'CAN_ID', 'DLC', 'DATA_X'])

    # Save DataFrame to CSV
    df.to_csv(output_file, index=False)

# Example usage
# convert_to_csv('input_data.txt', 'parsed_can_data.csv')


if __name__ == "__main__":
    for sub_dataset in ['OpelAstra', 'Prototype', 'RenaultClio']:
        print(sub_dataset)
        for log_file in os.listdir(os.path.join("../data/automotive_can_v2", sub_dataset)):
            if log_file.endswith(".log"):
                print(log_file)
                convert_to_csv(
                    os.path.join("../data/automotive_can_v2",
                                 sub_dataset, log_file),
                    os.path.join("../data/automotive_can_v2", sub_dataset,
                                 log_file.split(".")[0]+".csv")
                )
                print()
