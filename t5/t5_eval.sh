#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <test_filename> <finetuned_model_dir>"
    exit 1
fi

# Assign arguments to variables

TEST_FILE=$1
FINE_TUNED_MODEL_DIR=$2

# Run the Python script with the provided filenames
python3 t5_eval.py "$TEST_FILE" "$FINE_TUNED_MODEL_DIR"

