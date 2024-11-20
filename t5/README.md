1. Install dependencies
pip install transformers torch scikit-learn pandas tqdm numpy

2. **Fine-Tune the T5 Model: 
sh t5_finetune.sh <train_filename> <val_filename> <base_model> <finetuned_model_dir>**
Examples: 
a) Finetune t5 on german
--> sh t5_finetune.sh dataset/german/german_reviews_train.csv dataset/german/german_reviews_val.csv t5 t5_german_model
b) Finetune the above german t5 model on spanish
--> sh t5_finetune.sh dataset/spanish/spanish_reviews_train.csv dataset/spanish/spanis_reviews_val.csv t5_german_model t5_german_spanish_model

3. **Evaluate the T5 Model:
--> bash t5_eval.sh <test_filename> <finetuned_model_dir>**
Example: 
a) Evalute t5_german_model on german
--> bash t5_eval.sh dataset/german/german_test_train.csv t5_german_model


