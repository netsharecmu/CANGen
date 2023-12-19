import os
import time
import subprocess

# Import product from itertools
from itertools import product

test_train_files = {}
carhacking_test_files = [os.path.join("../data_selected/car_hacking", f)
                         for f in os.listdir("../data_selected/car_hacking") if "test.csv" in f]
for test_file in carhacking_test_files:
    test_category = os.path.basename(test_file).split("_")[0].lower()
    test_train_files[test_file] = []
    # Add real train files
    test_train_files[test_file].append(test_file.replace("test", "train"))

    # Add synthetic train files
    for f in os.listdir("../results/vehiclesec2024/small-scale/csv_selected"):
        if 'car-hacking' in f and test_category in f:
            test_train_files[test_file].append(os.path.join(
                "../results/vehiclesec2024/small-scale/csv_selected", f))

    assert len(test_train_files[test_file]) == 19  # 1 real + 3 runs * 6 models

ad_models = ['decision_tree', 'logistic_regression',
             'naive_bayes', 'mlp', 'random_forest', ]

os.makedirs("carhacking_ad_jobs", exist_ok=True)
# Iterate over all combinations of train and test files and od models (use product)
# Create a job for each combination
n_jobs = 0
for test_file, train_files in test_train_files.items():
    for train_file in train_files:
        for ad_model in ad_models:
            # print(train_file, test_file, od_model)
            # Create a job for each combination
            job_name = f"ad,{os.path.splitext(os.path.basename(train_file))[0]},{os.path.splitext(os.path.basename(test_file))[0]},{ad_model}"
            # print(job_name)
            results_json_file = f"../results/vehiclesec2024/small-scale/carhacking_ad_results/{job_name}.json"

            if os.path.exists(results_json_file):
                continue

            print(job_name)
            with open(os.path.join("carhacking_ad_jobs", job_name+".job"), 'w') as f:
                f.write(f"#!/bin/bash\n")
                f.write(f"#SBATCH -N 1\n")
                f.write(f"#SBATCH -p RM-shared\n")
                f.write(f"#SBATCH --ntasks-per-node=64\n")
                f.write(f"#SBATCH -t 02:00:00\n")
                f.write(f"#SBATCH -A cie160013p\n")
                f.write(f"#SBATCH --mail-type=ALL\n\n")

                f.write("set -x\n\n")
                f.write("cd /ocean/projects/cis230033p/yyin4/CANGen/eval\n")
                f.write("module load cuda\n")
                f.write("module load anaconda3\n")
                f.write("conda activate CANGen\n\n")

                f.write(
                    f"python3 carhacking-ad.py --train_csv_path {train_file} --test_csv_path {test_file} --results_json_file {results_json_file} --model_type {ad_model}\n")
                f.close()

            time.sleep(1)

            # Run the job using sbatch
            cmd = f"sbatch \
                -o carhacking_ad_jobs/{job_name}.out \
                -e carhacking_ad_jobs/{job_name}.out \
                --job-name={job_name} \
                carhacking_ad_jobs/{job_name}.job"
            subprocess.Popen(cmd, shell=True)
            time.sleep(2)

            n_jobs += 1
            print(f"Currently created {n_jobs} jobs")

print("Number of jobs created: ", n_jobs)
