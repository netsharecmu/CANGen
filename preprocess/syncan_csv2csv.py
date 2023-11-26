import os

import pandas as pd


def process_csv(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Convert the 'ID' column
    df['ID'] = df['ID'].apply(lambda x: int(x.lstrip('id')))

    # Write the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    for csv_file in os.listdir("../data/syncan/raw"):
        if csv_file.endswith(".csv"):
            print(csv_file)
            process_csv(
                os.path.join("../data/syncan/raw", csv_file),
                os.path.join("../data/syncan", csv_file)
            )
            print()
