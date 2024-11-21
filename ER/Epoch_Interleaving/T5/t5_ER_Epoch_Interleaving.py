import numpy as np
import pandas as pd
import time
import datetime
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def load_data(filename,rows):
    df = pd.read_csv(filename, nrows=rows)
    return df

def preprocess_data(df):
    reviews = df['review_body']
    labels = df['sentiment_label']
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return reviews, labels

def tokenize_reviews(reviews, labels, tokenizer, instruction, max_len=512):
    input_ids = []
    attention_masks = []
    decoder_input_ids = []
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    labels_text = [label_map[label] for label in labels]
    
    for review, label in zip(reviews, labels_text):
        review_with_instruction = f"{instruction} {review}"
        encoded_dict = tokenizer.encode_plus(
            review_with_instruction,
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
    
def create_dataloaders(train_dataset, dev_dataset, batch_size):
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


def train_model(model, optimizer, scheduler, first_dataloader, second_dataloader, val_dataloader, total_epochs=12):
    total_t0 = time.time()
    training_stats = []

    for epoch_i in range(total_epochs):
        # Determine which dataloader to use based on epoch
        if (epoch_i + 1) % 4 == 0:
            print(f'\n======== Epoch {epoch_i + 1} / {total_epochs} ========')
            print('Training on French data...')
            current_dataloader = first_dataloader
        else:
            print(f'\n======== Epoch {epoch_i + 1} / {total_epochs} ========')
            print('Training on English data...')
            current_dataloader = second_dataloader

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for batch in tqdm(current_dataloader):
            b_input_ids, b_input_mask, b_decoder_input_ids = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            output = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_decoder_input_ids)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(current_dataloader)
        print(f" Average training loss: {avg_train_loss:.2f}")

        # Validation step remains unchanged
        print("\nRunning Validation...")
        model.eval()
        total_eval_loss = 0

        for batch in tqdm(val_dataloader):
            b_input_ids, b_input_mask, b_decoder_input_ids = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                output = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_decoder_input_ids)
                loss = output.loss
                total_eval_loss += loss.item()

        avg_val_loss = total_eval_loss / len(val_dataloader)
        print(f" Validation Loss: {avg_val_loss:.2f}")

        training_stats.append({
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss
        })

    print("\nTraining complete!")
    print(f"Total training took {format_time(time.time() - total_t0)}")
    return training_stats

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
        generated_ids = model.generate(input_ids=b_input_ids, attention_mask=b_input_mask, max_length=b_decoder_input_ids.size(1))
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        true_labels = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_decoder_input_ids]
        for pred, true_label in zip(preds, true_labels):
            if pred.strip() == true_label.strip():
                total_eval_accuracy += 1

    avg_eval_accuracy = total_eval_accuracy / len(dataloader.dataset)
    avg_eval_loss = total_eval_loss / len(dataloader)
    return avg_eval_accuracy

def load_model(model_dir):    
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    
    model.to(device) 
    return model, tokenizer

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)  
    tokenizer.save_pretrained(output_dir)

def main():
    if len(sys.argv) != 6:
        print("Usage: python3 finetune.py <first_train> <first_val> <second_train> <second_val> <base_model> <finetuned_model_dir>")
        sys.exit(1)

    first_train_filename = sys.argv[1]
    first_val_filename = sys.argv[2]
    second_train_filename = sys.argv[3]
    second_val_filename = sys.argv[4]

    base_model = sys.argv[5]
    finetuned_model_dir = sys.argv[6]

    if(base_model=="t5"):
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        model.to(device)
    else: 
        model, tokenizer = load_model(base_model)

    # Specify the number of rows to limit the training size; use None to load the entire dataset.
    first_train_data = load_data(first_train_filename, 5000)
    second_train_data = load_data(second_train_filename, None)
    second_val_data = load_data(second_val_filename, None)

    first_reviews_train, first_labels_train = preprocess_data(first_train_data)
    second_reviews_train, second_labels_train = preprocess_data(second_train_data)
    second_reviews_val, second_labels_val = preprocess_data(second_val_data)


    instruction = "Classify the sentiment of the review as positive or negative or neutral:"
    first_input_ids_train , first_attention_mask_train , first_decoder_input_ids_train = tokenize_reviews(first_reviews_train ,first_labels_train ,tokenizer,instruction)
    second_input_ids_train , second_attention_mask_train , second_decoder_input_ids_train = tokenize_reviews(second_reviews_train ,second_labels_train ,tokenizer,instruction)
    val_input_ids_second_val , val_attention_mask_second_val , val_decoder_input_ids_second_val = tokenize_reviews(second_reviews_val ,second_labels_val ,tokenizer,instruction)
    
    first_dataset_training= TensorDataset(first_input_ids_train ,first_attention_mask_train ,first_decoder_input_ids_train )
    second_dataset_training= TensorDataset(second_input_ids_train ,second_attention_mask_train ,second_decoder_input_ids_train )
    second_dataset_validation= TensorDataset(val_input_ids_second_val,val_attention_mask_second_val,val_decoder_input_ids_second_val )

    first_dataloader_training, _ = create_dataloaders(first_dataset_training, first_dataset_training, batch_size=64)
    second_dataloader_training, second_dataloader_validation = create_dataloaders(second_dataset_training, second_dataset_validation, batch_size=64)
    
    epochs = 5
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
    total_steps = (len(first_dataloader_training) + len(second_dataloader_training))* epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    training_stats = train_model(model, optimizer, scheduler, first_dataloader_training, second_dataloader_training,second_dataloader_validation, epochs)
    finetuned_model_dir = "./" + finetuned_model_dir
    save_model(model, tokenizer, finetuned_model_dir)

if __name__ == "__main__":
    main()