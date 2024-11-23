#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <first_lang_train_filename> <first_lang_val_filename> <second_lang_train_filename> <second_lang_val_filename> <base_model> <finetuned_model_dir>"
    exit 1
fi

# Assign arguments to variables
FIRST_TRAIN_FILE=$1
FIRST_VAL_FILE=$2
SECOND_TRAIN_FILE=$3
SECOND_VAL_FILE=$4
BASE_MODEL=$5 #If default then use t5
FINE_TUNED_MODEL_DIR=$6

python3 t5_ER_Batch_Interleaving.py "$FIRST_TRAIN_FILE" "$FIRST_VAL_FILE" "$SECOND_TRAIN_FILE" "$SECOND_VAL_FILE" "$BASE_MODEL" "$FINE_TUNED_MODEL_DIR" 