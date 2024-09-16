import os

from car_challenge_csv2csv import convert_to_csv

if __name__ == "__main__":
    print("Idling.csv")
    convert_to_csv(
        input_file="../data/ttids/raw/Idling.csv",
        output_file="../data/ttids/Idling.csv"
    )

    for csv_file in os.listdir("../data/ttids/raw/MasqueradeAttack_BusOff"):
        if csv_file.endswith(".csv"):
            print(csv_file)
            convert_to_csv(
                os.path.join(
                    "../data/ttids/raw/MasqueradeAttack_BusOff", csv_file),
                os.path.join("../data/ttids", csv_file)
            )
            print()

    for csv_file in os.listdir("../data/ttids/raw/MasqueradeAttack_UDS"):
        if csv_file.endswith(".csv"):
            print(csv_file)
            convert_to_csv(
                os.path.join(
                    "../data/ttids/raw/MasqueradeAttack_UDS", csv_file),
                os.path.join("../data/ttids", csv_file)
            )
            print()
