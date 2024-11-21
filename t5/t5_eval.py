from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
import pandas as pd
import time
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def preprocess_data(df):
    reviews = df['review_body']
    labels = df['sentiment_label']
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return reviews, labels

def tokenize_reviews(reviews, labels, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []
    decoder_input_ids = []
    
    # Convert labels to text, as T5 expects text inputs
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    labels_text = [label_map[label] for label in labels]
    
    for review, label in zip(reviews, labels_text):
        encoded_dict = tokenizer.encode_plus(
            review,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

        target_ids = tokenizer.encode(
            label,
            max_length=10,  
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        decoder_input_ids.append(target_ids)
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    decoder_input_ids = torch.cat(decoder_input_ids, dim=0)
    
    return input_ids, attention_masks, decoder_input_ids

def load_model(model_dir):
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Send model to GPU if available
    return model, tokenizer

def evaluate(model, dataloader, tokenizer):
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in tqdm(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_decoder_input_ids = batch[2].to(device)

        with torch.no_grad():
            output = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_decoder_input_ids)
        loss = output.loss
        total_eval_loss += loss.item()

        # Generate sequences without teacher forcing (to compare the predictions)
        generated_ids = model.generate(input_ids=b_input_ids, attention_mask=b_input_mask, max_length=b_decoder_input_ids.size(1))
        # Decode both predicted and true sequences for comparison
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        true_labels = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_decoder_input_ids]
        for pred, true_label in zip(preds, true_labels):
            if pred.strip() == true_label.strip():
                total_eval_accuracy += 1

    avg_eval_accuracy = total_eval_accuracy / len(dataloader.dataset)
    avg_eval_loss = total_eval_loss / len(dataloader)
    return avg_eval_accuracy

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 t5_eval.py <test_filename> <finetuned_model_dir>")
        sys.exit(1)

    test_filename = sys.argv[1]
    model_dir = sys.argv[2]
    test = load_data(test_filename)

    model, tokenizer = load_model(model_dir)
    
    test_reviews, test_labels = preprocess_data(test)
    test_input_ids, test_attention_mask, test_decoder_input_ids = tokenize_reviews(test_reviews, test_labels, tokenizer)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_decoder_input_ids)    

    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=64
    )
    test_stats = evaluate(model, test_dataloader, tokenizer)
    print("Test stats", test_stats)
    
if __name__ == "__main__":
    main()
