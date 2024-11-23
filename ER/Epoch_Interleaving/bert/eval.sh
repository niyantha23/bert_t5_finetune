#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <finetuned_model_dir> <test_filename> <lang_name>"
    exit 1
fi

# Assign arguments to variables

FINE_TUNED_MODEL_DIR=$1
TEST_FILE=$2
LANG_NAME=$3

# Run the Python script with the provided filenames
python3 eval.py "$FINE_TUNED_MODEL_DIR" "../../../$TEST_FILE" "$LANG_NAME"