#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <train_filename> <val_filename> <test_filename>"
    exit 1
fi

# Assign arguments to variables
TRAIN_FILE=$1
VAL_FILE=$2
TEST_FILE=$3

# Run the Python script with the provided filenames
python3 finetune.py "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"
python finetune_bert.py "english/english_reviews_train.csv" "english/english_reviews_val.csv" "english/english_reviews_test.csv" "bert_model_eng"