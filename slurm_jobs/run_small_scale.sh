#!/bin/bash

# Example usage (model and dataset names are separated by "--"):
#     bash run_small_scale.sh -p small-scale model1 model2 model3 -- dataset1 dataset2 dataset3

# Initialize the config_partition variable
config_partition=""

# Parse the options
while getopts "p:" opt; do
  case $opt in
    p) config_partition="$OPTARG";;
    \?) echo "Invalid option: -$OPTARG" >&2
        exit 1;;
    :) echo "Option -$OPTARG requires an argument." >&2
       exit 1;;
  esac
done

# Shift out the options
shift $((OPTIND-1))

# Check if the config_partition is set, if not exit
if [ -z "$config_partition" ]; then
    echo "Please provide the config_partition using -p option."
    exit 1
fi

# Separate the input arguments into model names and dataset names
sep_flag=0
model_names=()
dataset_names=()
for arg in "$@"; do
    if [ "$arg" == "--" ]; then
        sep_flag=1
        continue
    fi
    if [ $sep_flag -eq 0 ]; then
        model_names+=("$arg")
    else
        dataset_names+=("$arg")
    fi
done

task_count=0
runs_per_job=3 # specify how many times to run each job


mkdir -p small_scale
for ((i=1; i<=runs_per_job; i++)); do
    for model in "${model_names[@]}"; do
        for dataset in "${dataset_names[@]}"; do
            timestamp=$(date +%Y%m%d%H%M%S%N) # %N for nanoseconds
            job_name="${config_partition}-${model}-${dataset}-${timestamp}"
            cmd="sbatch \
            -o small_scale/${job_name}.out \
            -e small_scale/${job_name}.out \
            --export=config_partition=${config_partition},dataset_name=${dataset},model_name=${model},cur_time=${timestamp} \
            --job-name=${job_name} \
            single_driver_rtf_naive.job"

            echo $cmd
            eval $cmd

            sleep 1 # pause to be kind to the scheduler

            # Increment task counter
            ((task_count++))
        done
    done
done

# Debug: Print total number of tasks
echo "Total number of tasks: $task_count"