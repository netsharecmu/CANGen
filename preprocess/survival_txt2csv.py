import os

from car_hacking_attack_csv2csv import convert_csv

if __name__ == "__main__":
    for sub_folder in ['Sonata', 'Soul', 'Spark']:
        for txt_file in os.listdir(os.path.join("../data/survival", sub_folder)):
            if txt_file.endswith(".txt"):
                print(txt_file)
                convert_csv(
                    input_csv=os.path.join(
                        "../data/survival", sub_folder, txt_file),
                    output_csv=os.path.join(
                        "../data/survival", sub_folder, txt_file.split(".")[0]+".csv")
                )
                print()
