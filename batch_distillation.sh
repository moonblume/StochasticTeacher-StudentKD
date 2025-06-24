#!/bin/bash

# For conda: set the path to the anaconda environment
export PATH=$PATH: # path to the anaconda environment

# Set PYTHONPATH to include the src directory
export PYTHONPATH=$PYTHONPATH: # path to the src directory

# Base directory containing all fold directories
base_dir= "path to the base directory containing all fold directories"

# Directory to save results
results_dir="path to save results"
mkdir -p $results_dir

echo '========= Running knowledge distillation ========='
echo "Results will be saved in $results_dir"
echo "Base directory: $base_dir"

# Path to the config file
config_file="path to /src/config.json"

# Loop through each fold directory and find the model_best.pth file
for fold_dir in $base_dir/*; do
    if [ -d "$fold_dir" ]; then
        # Extract the fold ID from the directory name
        echo "Processing fold directory: $fold_dir"
        fold_id=$(basename "$fold_dir" | grep -oP '(?<=fold)\d+$')
        echo "Fold ID: $fold_id"
        # Ensure fold_id is an integer
        if ! [[ "$fold_id" =~ ^[0-9]+$ ]]; then
            echo "Invalid fold ID: $fold_id. Skipping."
            continue
        fi

        teacher_model_path=$(find "$fold_dir" -name "model_best.pth")

        if [ -z "$teacher_model_path" ]; then
            echo "No model_best.pth found in $fold_dir. Skipping."
            continue
        fi

        echo "Running distillation for fold ${fold_id} with teacher model at ${teacher_model_path}"

        /home/livia/anaconda3/envs/affwild/bin/python path to /Tdistillation.py \
            --config ${config_file} \
            --device 0 \
            --fold_id ${fold_id} \
            --teacher_model_path ${teacher_model_path} \
            --results_dir ${results_dir} | tee "${results_dir}/fold_${fold_id}_log.txt"

        if [ $? -ne 0 ]; then
            echo "Distillation failed for fold ${fold_id}. Skipping to next fold."
            continue
        fi
    fi
done

echo '========= Model has been trained with knowledge distillation ========='
