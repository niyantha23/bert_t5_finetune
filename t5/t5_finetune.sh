#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <train_filename> <val_filename> <base_model> <finetuned_model_dir>"
    exit 1
fi

# Assign arguments to variables
TRAIN_FILE=$1
VAL_FILE=$2
BASE_MODEL=$3 #If default then use t5
FINE_TUNED_MODEL_DIR=$4

python3 t5_finetune.py "$TRAIN_FILE" "$VAL_FILE" "$BASE_MODEL" "$FINE_TUNED_MODEL_DIR" 