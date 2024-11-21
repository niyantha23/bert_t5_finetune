## Install dependencies<br/>
pip install transformers torch scikit-learn pandas tqdm numpy

## Fine-Tune the T5 Model 
sh t5_finetune.sh <train_filename> <val_filename> <base_model> <finetuned_model_dir>

Examples: <br/>
1. Finetune on german:<br/>
   sh t5_finetune.sh german_reviews_train.csv german_reviews_val.csv t5 t5_german_model <br/>
2. Finetune the above model on spanish:<br/>
   sh t5_finetune.sh spanish_reviews_train.csv spanis_reviews_val.csv t5_german_model t5_german_spanish_model <br/>

## Evaluate the T5 Model
bash t5_eval.sh <test_filename> <finetuned_model_dir>
   
Example: <br/>
1. Evalute t5_german_model on german: bash t5_eval.sh german_test_train.csv t5_german_model


