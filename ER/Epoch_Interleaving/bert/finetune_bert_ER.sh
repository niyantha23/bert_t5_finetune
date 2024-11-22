#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 10 ]; then
    echo "Usage: $0 <base_lang> <first_train_filename> <first_val_filename> <first_test_filename> <finetune_lang> <second_train_filename> <second_val_filename> <second_test_filename> <base_model_name> <outname>"
    exit 1
fi

# Assign arguments to variables
BASE_LANG=$1
FIRST_TRAIN_FILE=$2
FIRST_VAL_FILE=$3
FIRST_TEST_FILE=$4
FINETUNE_LANG=$5
SECOND_TRAIN_FILE=$6
SECOND_VAL_FILE=$7
SECOND_TEST_FILE=$8
BASE_MODEL_FILE=$9
FINETUNED_MODEL_OUT=$10

# Run the Python script with the provided arguments
python3 finetune_bert_ER.py \
    --base_lang "$BASE_LANG" \
    --first_train_filename "../../../$FIRST_TRAIN_FILE" \
    --first_val_filename "../../../$FIRST_VAL_FILE" \
    --first_test_filename "../../../$FIRST_TEST_FILE" \
    --finetune_lang "$FINETUNE_LANG" \
    --second_train_filename "../../../$SECOND_TRAIN_FILE" \
    --second_val_filename "../../../$SECOND_VAL_FILE" \
    --second_test_filename "../../../$SECOND_TEST_FILE" \
    --base_model_file "$BASE_MODEL_FILE" \
    --finetuned_model_path "$FINETUNED_MODEL_OUT"