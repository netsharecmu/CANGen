import os
import time
import subprocess

# Import product from itertools
from itertools import product

syncan_test_files = [os.path.join("../data_selected/syncan", f)
                     for f in os.listdir("../data_selected/syncan") if f.startswith("test_")]

print(syncan_test_files)


synthetic_train_files = []
for f in os.listdir("../results/vehiclesec2024/small-scale/csv"):
    if 'syncan' in f and '2023' in f:
        synthetic_train_files.append(os.path.join(
            "../results/vehiclesec2024/small-scale/csv", f))
synthetic_train_files.append("../data_selected/syncan/train.csv")

print(len(synthetic_train_files))

od_models = ['ocsvm', 'iforest', 'lof', 'kmeans', 'dbscan']

os.makedirs("syncan_od_jobs", exist_ok=True)
# Iterate over all combinations of train and test files and od models (use product)
# Create a job for each combination
n_jobs = 0
for train_file, test_file, od_model in product(synthetic_train_files, syncan_test_files, od_models):
    # print(train_file, test_file, od_model)
    # Create a job for each combination
    job_name = f"od,{os.path.splitext(os.path.basename(train_file))[0]},{os.path.splitext(os.path.basename(test_file))[0]}_{od_model}"
    print(job_name)
    with open(os.path.join("syncan_od_jobs", job_name+".job"), 'w') as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"#SBATCH -N 1\n")
        f.write(f"#SBATCH -p RM-shared\n")
        f.write(f"#SBATCH --ntasks-per-node=16\n")
        f.write(f"#SBATCH -t 02:00:00\n")
        f.write(f"#SBATCH -A cie160013p\n")
        f.write(f"#SBATCH --mail-type=ALL\n\n")

        f.write("set -x\n\n")
        f.write("cd /ocean/projects/cis230033p/yyin4/CANGen/eval\n")
        f.write("module load cuda\n")
        f.write("module load anaconda3\n")
        f.write("conda activate CANGen\n\n")

        results_json_file = f"../results/vehiclesec2024/small-scale/syncan_od_results/{job_name}.json"

        f.write(
            f"python3 syncan-outlier-detection-sklearn.py --train_csv_path {train_file} --test_csv_path {test_file} --results_json_file {results_json_file} --model_type {od_model} --sample_size 0.1\n")
        f.close()

    # Run the job using sbatch
    cmd = f"sbatch \
        -o syncan_od_jobs/{job_name}.out \
        -e syncan_od_jobs/{job_name}.out \
        --job-name={job_name} \
        syncan_od_jobs/{job_name}.job"
    subprocess.Popen(cmd, shell=True)
    time.sleep(1)

    n_jobs += 1
    print(f"Currently created {n_jobs} jobs")

print("Number of jobs created: ", n_jobs)
