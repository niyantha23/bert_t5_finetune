import numpy as np
import pandas as pd
import time
import datetime
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def load_data(filename, rows):
    df = pd.read_csv(filename, nrows=rows)
    return df

def preprocess_data(df):
    reviews = df['review_body']
    labels = df['sentiment_label']
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    return reviews, labels

def save_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)  
    tokenizer.save_pretrained(output_dir)

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

def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def create_interleaved_dataloader(first_dataset, second_dataset, batch_size, first_ratio=0.125):
    first_size = int(batch_size * first_ratio)  # Number of samples from the first dataset
    second_size = batch_size - first_size  # Remaining samples from the second dataset

    first_dataloader = DataLoader(first_dataset, sampler=RandomSampler(first_dataset), batch_size=first_size)
    second_dataloader = DataLoader(second_dataset, sampler=RandomSampler(second_dataset), batch_size=second_size)

    def interleaved_batches():
        first_iter = iter(first_dataloader)
        second_iter = iter(second_dataloader)

        while True:
            try:
                first_batch = next(first_iter)
                second_batch = next(second_iter)
                yield (
                    torch.cat((first_batch[0], second_batch[0]), dim=0),
                    torch.cat((first_batch[1], second_batch[1]), dim=0),
                    torch.cat((first_batch[2], second_batch[2]), dim=0),
                )
            except StopIteration:
                break

    # Calculate total batches in advance
    total_batches = min(len(first_dataloader), len(second_dataloader))
    return interleaved_batches(), total_batches
    
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

def create_mixed_validation_dataset(first_val_dataset, second_val_dataset, ratio_first=0.3, ratio_second=0.7):
    first_val_size = min(len(first_val_dataset), int(len(first_val_dataset) * ratio_first))
    second_val_size = min(len(second_val_dataset), int(len(second_val_dataset) * ratio_second))

    mixed_first_val_data = random.sample(list(first_val_dataset), first_val_size)
    mixed_second_val_data = random.sample(list(second_val_dataset), second_val_size)

    mixed_val_data = mixed_first_val_data + mixed_second_val_data
    random.shuffle(mixed_val_data)

    mixed_input_ids = []
    mixed_attention_masks = []
    mixed_decoder_input_ids = []

    for x in mixed_val_data:
        mixed_input_ids.append(x[0].unsqueeze(0))
        mixed_attention_masks.append(x[1].unsqueeze(0))
        mixed_decoder_input_ids.append(x[2].unsqueeze(0))

    mixed_input_ids = torch.cat(mixed_input_ids, dim=0)
    mixed_attention_masks = torch.cat(mixed_attention_masks, dim=0)
    mixed_decoder_input_ids = torch.cat(mixed_decoder_input_ids, dim=0)

    mixed_val_dataset = TensorDataset(mixed_input_ids, mixed_attention_masks, mixed_decoder_input_ids)
    return mixed_val_dataset

def train_model(model, optimizer, scheduler, first_dataset, second_dataset, first_valdataloader, second_valdataloader, total_epochs=12):
    total_t0 = time.time()
    training_stats = []

    for epoch_i in range(total_epochs):
        print(f'\n======== Epoch {epoch_i + 1} / {total_epochs} ========')
        print('Training with interleaved batches...')
        
        t0 = time.time()
        total_train_loss = 0
        model.train()
        interleaved_batches, total_batches = create_interleaved_dataloader(first_dataset, second_dataset, batch_size =64, first_ratio=0.125)

        for batch in tqdm(interleaved_batches, total=total_batches):
            b_input_ids, b_input_mask, b_decoder_input_ids = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            optimizer.zero_grad()
            output = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_decoder_input_ids)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / total_batches 
        print(f" Average training loss: {avg_train_loss:.2f}")

        # Validation step remains unchanged
        print("\nRunning Validation...")
        model.eval()
        total_eval_loss = 0
        mixed_val_dataset = create_mixed_validation_dataset(first_valdataloader.dataset, second_valdataloader.dataset)
        mixed_val_dataloader = DataLoader(mixed_val_dataset, sampler=SequentialSampler(mixed_val_dataset), batch_size=64)
        val_dataloader = mixed_val_dataloader

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

def main():
    if len(sys.argv) != 7:
        print("Usage: python t5_ER_Batch_Interleaving.py <first_lang_train_filename> <first_lang_val_filename> <second_lang_train_filename> <second_lang_val_filename> <base_model> <finetuned_model_dir>")
        sys.exit(1)

    first_train_filename = sys.argv[1]
    first_val_filename = sys.argv[2]
    second_train_filename = sys.argv[3]
    second_val_filename = sys.argv[4]
    base_model_dir = sys.argv[5]
    finetuned_model_dir = sys.argv[6]
    
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.to(device)

    # Load and preprocess datasets for both languages
    first_train_data = load_data(first_train_filename, None)
    second_train_data = load_data(second_train_filename, None)
    first_val_data = load_data(first_val_filename, None)
    second_val_data = load_data(second_val_filename, None)

    first_reviews_train, first_labels_train = preprocess_data(first_train_data)
    second_reviews_train, second_labels_train = preprocess_data(second_train_data)
    first_reviews_val, first_labels_val = preprocess_data(first_val_data)
    second_reviews_val, second_labels_val = preprocess_data(second_val_data)
    
    instruction = "Classify the sentiment of the review as positive or negative or neutral:"
    
    first_input_ids_train , first_attention_mask_train , first_decoder_input_ids_train = tokenize_reviews(first_reviews_train ,first_labels_train ,tokenizer,instruction)
    second_input_ids_train , second_attention_mask_train , second_decoder_input_ids_train = tokenize_reviews(second_reviews_train ,second_labels_train ,tokenizer,instruction)
    val_input_ids_first_val , val_attention_mask_first_val , val_decoder_input_ids_first_val = tokenize_reviews(first_reviews_val ,first_labels_val ,tokenizer,instruction)
    val_input_ids_second_val , val_attention_mask_second_val , val_decoder_input_ids_second_val = tokenize_reviews(second_reviews_val ,second_labels_val ,tokenizer,instruction)

    first_dataset_training= TensorDataset(first_input_ids_train ,first_attention_mask_train ,first_decoder_input_ids_train )
    second_dataset_training= TensorDataset(second_input_ids_train ,second_attention_mask_train ,second_decoder_input_ids_train )
    first_dataset_validation= TensorDataset(val_input_ids_first_val,val_attention_mask_first_val,val_decoder_input_ids_first_val)
    second_dataset_validation= TensorDataset(val_input_ids_second_val,val_attention_mask_second_val,val_decoder_input_ids_second_val )


    first_dataloader_training, first_dataloader_validation = create_dataloaders(first_dataset_training, first_dataset_validation, batch_size=64)
    second_dataset_validation= TensorDataset(val_input_ids_second_val,val_attention_mask_second_val,val_decoder_input_ids_second_val)
    _, second_dataloader_validation=create_dataloaders(second_dataset_validation ,second_dataset_validation,batch_size=64)

    epochs = 5 
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
    # total_steps = total_batches * epochs
    # print(total_steps)
    total_steps = 7810
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    training_stats = train_model(model, optimizer, scheduler, first_dataset_training, second_dataset_training, first_dataloader_validation, second_dataloader_validation, epochs)

    finetuned_model_dir = "./" + finetuned_model_dir
    save_model(model, tokenizer, finetuned_model_dir)

if __name__=="__main__":
    main() 