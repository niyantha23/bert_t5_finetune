import numpy as np
import pandas as pd
import time
import datetime
import random
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# Load dataset
def load_data(filename):
    df = pd.read_csv(filename)
    return df

# Preprocess data
def preprocess_data(df):
    reviews = df['review_body']
    labels = df['sentiment_label']
    
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    
    return reviews,labels

# Tokenize reviews
def tokenize_reviews(reviews, labels, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []
    
    for review in reviews:
        encoded_dict = tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    
    return input_ids, attention_masks, labels

def evaluate(model,dataloader,device):
    predictions = []
    total_eval_accuracy=0
    for batch in tqdm(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels=batch[2].to(device)
        with torch.no_grad():        
            output= model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            #pred_flat = np.argmax(logits, axis=1).flatten()
            #predictions.extend(list(pred_flat))
            #print(logits,predictions)
            label_ids=b_labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)
    avg_test_accuracy = total_eval_accuracy / len(dataloader)
    return avg_test_accuracy

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))
def main():
    # Load and preprocess data
    if len(sys.argv) != 4:
        print("Usage: python3 finetune.py <model_name> <test_filename> <lang_name>")
        sys.exit(1)

    model_name = sys.argv[1]
    test_filename = sys.argv[2]
    lang_name = sys.argv[3]

    print(f"Running Eval: {lang_name}")

    test=load_data(test_filename)

    test_reviews, test_labels =preprocess_data(test)
    # Load tokenizer and tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=True)
    test_input_ids, test_attention_mask, test_labels = tokenize_reviews(test_reviews, test_labels, tokenizer)
    # Create datasets
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    # Create dataloaders
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=64
    )
    model = torch.load(model_name)
    print("Model loaded")
    test_stats=evaluate(model,test_dataloader,device)
    print(f"{lang_name} TEST STATS: {test_stats}")

if __name__ == "__main__":
    main()