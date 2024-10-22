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

# Create DataLoader
def create_dataloaders(train_dataset, dev_dataset, batch_size=32):
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )
    
    val_dataloader = DataLoader(
        dev_dataset,
        sampler=SequentialSampler(dev_dataset),
        batch_size=batch_size
    )
    
    return train_dataloader, val_dataloader

# Train the model
def train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs=4):
    total_t0 = time.time()
    training_stats = []
    
    for epoch_i in range(0, epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        
        for batch in tqdm(train_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_time = format_time(time.time() - t0)
        print(f"  Average training loss: {avg_train_loss:.2f}")
        print(f"  Training epoch took: {training_time}")
        
        print("\nRunning Validation...")
        t0 = time.time()
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        best_eval_accuracy = 0
        
        for batch in tqdm(val_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_eval_loss += loss.item()
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        validation_time = format_time(time.time() - t0)
        #scheduler.step(avg_val_loss)
        if avg_val_accuracy > best_eval_accuracy:
            torch.save(model, 'bert_model_eng')
            best_eval_accuracy = avg_val_accuracy
        
        print(f"  Accuracy: {avg_val_accuracy:.2f}")
        print(f"  Validation Loss: {avg_val_loss:.2f}")
        print(f"  Validation took: {validation_time}")
        
        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        })
    
    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)}")
    return training_stats


def evaluate(model,dataloader):
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

# Utility functions
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Main function
def main():
    # Load and preprocess data
    if len(sys.argv) != 4:
        print("Usage: python3 finetune.py <train_filename> <val_filename> <test_filename>")
        sys.exit(1)

    # train_filename='"english/english_reviews_train.csv"'
    # val_filename='english/english_reviews_val.csv'
    # test_filename='english/english_reviews_test.csv'
    train_filename = sys.argv[1]
    val_filename = sys.argv[2]
    test_filename = sys.argv[3]
    train = load_data(train_filename)
    val=load_data(val_filename)
    test=load_data(test_filename)
    train_reviews, train_labels = preprocess_data(train)
    val_reviews, val_labels =preprocess_data(val)
    test_reviews, test_labels =preprocess_data(test)
    # Load tokenizer and tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_input_ids, train_attention_mask, train_labels = tokenize_reviews(train_reviews, train_labels, tokenizer)
    dev_input_ids, dev_attention_mask, dev_labels = tokenize_reviews(val_reviews, val_labels, tokenizer)
    test_input_ids, test_attention_mask, test_labels = tokenize_reviews(test_reviews, test_labels, tokenizer)
    # Create datasets
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    dev_dataset = TensorDataset(dev_input_ids, dev_attention_mask, dev_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(train_dataset, dev_dataset,64)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=64
    )
    epochs=10
    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3,hidden_dropout_prob=0.2)
    model = model.to(device)
    
    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-6, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs # 4 epochs
    
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.2*total_steps, num_training_steps=total_steps)
    
    # Train the model
    training_stats = train_model(model, optimizer, scheduler, train_dataloader, val_dataloader,epochs)
    model = torch.load('bert_model_eng')
    test_stats=evaluate(model,test_dataloader)
    print(test_stats)
    return training_stats

if __name__ == "__main__":
    main()
