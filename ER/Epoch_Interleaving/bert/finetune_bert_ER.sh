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
FINETUNED_MODEL_OUT=${10}

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


python ER/Epoch_Interleaving/bert/Similarity/sim_test.py --base_lang fren
ch --first_train_filename french/french_reviews_train.csv --first_val_filename french/french_re
views_val.csv --first_test_filename french/french_reviews_test.csv --finetune_lang Spanish --se
cond_train_filename dataset/spanish/spanish_reviews_train.csv second_val_filename dataset/spani
sh/spanish_reviews_val.csv --second_test_filename dataset/spanish/spanish_reviews_test.csv --ba
se_model_file models/mbert_model_frn_5--finetuned_model_path /models2/simtest
