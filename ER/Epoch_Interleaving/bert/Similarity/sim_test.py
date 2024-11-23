import numpy as np
import pandas as pd
import time
import datetime
import random
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertTokenizer, get_linear_schedule_with_warmup
import sys
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset

def load_data(filename, rows):
    df = pd.read_csv(filename)

    if rows is None:
        return shuffle(df, random_state=42).reset_index(drop=True)

    unique_labels = df['sentiment_label'].unique()
    num_labels = len(unique_labels)
    rows_per_label = rows // num_labels
    balanced_df = pd.DataFrame()

    for label in unique_labels:
        label_df = df[df['sentiment_label'] == label]
        sampled_label_df = label_df.sample(n=rows_per_label, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled_label_df])

    balanced_df = shuffle(balanced_df, random_state=42).reset_index(drop=True)
    return balanced_df


def preprocess_data(df):
    reviews = df['review_body']
    labels = df['sentiment_label']

    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    return reviews, labels

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

# Train the model


def train_model(model, optimizer, scheduler, first_train_dataloader, second_train_dataloader, first_valdataloader, second_valdataloader, outname, epochs=12):
    total_t0 = time.time()
    training_stats = []
    
    for epoch_i in range(0, epochs):

        if (epoch_i + 1) % 4 == 0:
            print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
            print('Training on French data...')
            current_dataloader = first_train_dataloader
            mixed_val_dataset = create_mixed_validation_dataset(first_valdataloader.dataset, second_valdataloader.dataset)
            mixed_val_dataloader = DataLoader(mixed_val_dataset, sampler=SequentialSampler(mixed_val_dataset), batch_size=64)
            val_dataloader = mixed_val_dataloader
            
        else:
            print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
            print('Training on Spanish data...')
            current_dataloader = second_train_dataloader
            val_dataloader = second_valdataloader

        print(f'\n======== Epoch {epoch_i + 1} / {epochs} ========')
        print('Training...')

        t0 = time.time()
        total_train_loss = 0
        model.train()

        for batch in tqdm(current_dataloader):
            b_input_ids, b_input_mask, b_labels = batch[0].to(
                device), batch[1].to(device), batch[2].to(device)
            optimizer.zero_grad()
            output = model(
                b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(current_dataloader)
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
            b_input_ids, b_input_mask, b_labels = batch[0].to(
                device), batch[1].to(device), batch[2].to(device)
            with torch.no_grad():
                output = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output.loss
            total_eval_loss += loss.item()
            logits = output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        validation_time = format_time(time.time() - t0)
        # scheduler.step(avg_val_loss)
        if avg_val_accuracy > best_eval_accuracy:
            torch.save(model, outname)
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


def evaluate(model, dataloader):
    predictions = []
    total_eval_accuracy = 0
    for batch in tqdm(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            output = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
            logits = output.logits
            logits = logits.detach().cpu().numpy()
            # pred_flat = np.argmax(logits, axis=1).flatten()
            # predictions.extend(list(pred_flat))
            # print(logits,predictions)
            label_ids = b_labels.to('cpu').numpy()
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

def find_top_k_similar_dataset(model, dataset1, dataset2, tokenizer, device, top_k=100, batch_size=128, max_len=512):
    """
    Finds the top k most similar data points in dataset1 for all points in dataset2 combined, using cosine similarity, optimized for GPU.
    """

    model.eval()  # Set the model to evaluation mode
    model.to(device)

    def get_embeddings(dataset):
        """Generates embeddings for a dataset."""
        dataloader = DataLoader(dataset, batch_size=batch_size)
        embeddings = []
        for batch in tqdm(dataloader):
            input_ids, attention_masks = [t.to(device) for t in batch[:2]]  # Ignore labels if present
            with torch.no_grad():
                outputs = model.bert(input_ids, attention_mask=attention_masks)
            embeddings.append(outputs.pooler_output)  # Use pooled output as embedding
        return torch.cat(embeddings, dim=0)

    # Tokenize dataset1 and dataset2
    reviews1, labels1 = preprocess_data(dataset1)
    reviews2, labels2 = preprocess_data(dataset2)
    
    input_ids1, attention_masks1, labels1 = tokenize_reviews(reviews1, labels1, tokenizer, max_len)
    input_ids2, attention_masks2, labels2 = tokenize_reviews(reviews2, labels2, tokenizer, max_len)

    dataset1_tensor = TensorDataset(input_ids1, attention_masks1,labels1)
    dataset2_tensor = TensorDataset(input_ids2, attention_masks2,labels2)

    # Compute embeddings for dataset1 and dataset2
    dataset1_embeddings = get_embeddings(dataset1_tensor)  # [N1, hidden_dim]
    dataset2_embeddings = get_embeddings(dataset2_tensor)  # [N2, hidden_dim]

    # Normalize embeddings for cosine similarity computation
    dataset1_embeddings = torch.nn.functional.normalize(dataset1_embeddings, dim=1)
    dataset2_embeddings = torch.nn.functional.normalize(dataset2_embeddings, dim=1)

    # Compute pairwise cosine similarities (N2, N1)
    similarity_matrix = torch.mm(dataset2_embeddings, dataset1_embeddings.T)  # [N2, N1]

    # Sum the similarities across all points in dataset2 for each point in dataset1
    total_similarities = similarity_matrix.sum(dim=0)  # [N1], summed similarity for each point in dataset1

    # Get the top_k most similar points from dataset1
    top_k_similarities, top_k_indices = torch.topk(total_similarities, top_k, largest=True, sorted=False)

    # Extract the top_k most similar points from dataset1 based on the indices
    top_k_data = [dataset1_tensor[idx] for idx in top_k_indices.cpu().tolist()]

    # Create a new dataset with the top_k similar data points
    reviews = []
    attention_masks = []
    labels = []

    for input_ids, attention_mask, label in top_k_data:
        # Decode the review text
        review_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        reviews.append(review_text)
        attention_masks.append(attention_mask.cpu().numpy())  # Convert mask tensor to numpy
        labels.append(label.item())  # Convert label tensor to a scalar value

    # Create DataFrame
    df = pd.DataFrame({
        'review_body': reviews,
        'AttentionMask': attention_masks,
        'sentiment_label': labels
    })

    return df

def main():
    # Load and preprocess data
    parser = argparse.ArgumentParser(
        description='Fine-tune a BERT model for sequence classification.')

    # File paths
    parser.add_argument('--outname', type=str, required=True, help='Path to save the trained model.')

    parser.add_argument('--base_lang', type=str,
                        required=True, help='Base Language')
    parser.add_argument('--first_train_filename', type=str,
                        required=True, help='Path to the training dataset file (CSV).')
    parser.add_argument('--first_val_filename', type=str, required=True,
                        help='Path to the validation dataset file (CSV).')
    parser.add_argument('--first_test_filename', type=str,
                        required=True, help='Path to the test dataset file (CSV).')

    parser.add_argument('--finetune_lang', type=str,
                        required=True, help='Finetuning Language')
    parser.add_argument('--second_train_filename', type=str,
                        required=True, help='Path to the training dataset file (CSV).')
    parser.add_argument('--second_val_filename', type=str, required=True,
                        help='Path to the validation dataset file (CSV).')
    parser.add_argument('--second_test_filename', type=str,
                        required=True, help='Path to the test dataset file (CSV).')

    # Base Model
    parser.add_argument('--base_model_file', type=str, required=True,
                        help='Path to load the base model.')
    parser.add_argument('--finetuned_model_path', type=str,
                        required=True, help='Path to save fine tuned model')

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training and validation.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float,
                        default=2e-5, help='Learning rate for AdamW optimizer.')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Epsilon value for the AdamW optimizer.')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay for the optimizer.')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps for learning rate scheduler.')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum sequence length for tokenization.')

    args = parser.parse_args()


    # train_filename='"english/english_reviews_train.csv"'
    # val_filename='english/english_reviews_val.csv'
    # test_filename='english/english_reviews_test.csv'

    base_lang = args.base_lang
    outname = args.outname
    first_train_filename = args.first_train_filename
    first_val_filename = args.first_val_filename
    first_test_filename = args.first_test_filename

    finetuned_lang = args.finetune_lang
    second_train_filename = args.second_train_filename
    second_val_filename = args.second_val_filename
    second_test_filename = args.second_test_filename

    
    base_model = args.base_model_file
    finetuned_output_path = args.finetuned_model_path
    epochs = args.epochs
    # Load pre-trained BERT model
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3,hidden_dropout_prob=0.2)
    if base_model is not None:
        model = torch.load(base_model)
        # model = torch.load(base_model, map_location=torch.device('cpu'))
        print(f'Loading saved model from :{base_model}')

    else:
        print("No base model found or path not provided, loading BERT pre-trained model")
        # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased", num_labels=3, hidden_dropout_prob=0.2)

    model = model.to(device)
    # Load tokenizer and tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    first_train_data = load_data(first_train_filename, 5000)
    #first_train_data = load_data(first_train_filename, None)
    first_val_data = load_data(first_val_filename, None)
    #second_train_data = load_data(second_train_filename, None)
    second_train_data = load_data(second_train_filename, 5000)
    second_val_data = load_data(second_val_filename, None)
    second_test_data = load_data(second_test_filename, None)

    top_similar_dataset = find_top_k_similar_dataset(
    model=model,
    dataset1=first_train_data,  # Example first dataset
    dataset2=second_train_data,  # Example second dataset
    tokenizer=tokenizer,
    device=device,
    top_k=1000
    )
    

    #first_reviews_train, first_labels_train = preprocess_data(first_train_data)
    first_reviews_train, first_labels_train = preprocess_data(top_similar_dataset)
    first_reviews_val, first_labels_val = preprocess_data(first_val_data)
    second_reviews_train, second_labels_train = preprocess_data(second_train_data)
    second_reviews_val, second_labels_val = preprocess_data(second_val_data)
    second_reviews_test, second_labels_test = preprocess_data(second_test_data)

    

    # Tokenize both the datasets
    first_input_ids_train, first_attention_mask_train, first_decoder_input_ids_train = tokenize_reviews(first_reviews_train, first_labels_train, tokenizer)
    val_input_ids_first_val, val_attention_mask_first_val, val_decoder_input_ids_first_val = tokenize_reviews(first_reviews_val, first_labels_val, tokenizer)
    second_input_ids_train, second_attention_mask_train, second_decoder_input_ids_train = tokenize_reviews(second_reviews_train, second_labels_train, tokenizer)
    val_input_ids_second_val, val_attention_mask_second_val, val_decoder_input_ids_second_val = tokenize_reviews(second_reviews_val, second_labels_val, tokenizer)
    test_input_ids_second_test, test_attention_mask_second_test, test_decoder_input_ids_second_test = tokenize_reviews(second_reviews_val, second_labels_val, tokenizer)

    # Create datasets
    first_dataset_training = TensorDataset(
        first_input_ids_train, first_attention_mask_train, first_decoder_input_ids_train)
    first_dataset_validation = TensorDataset(
        val_input_ids_first_val, val_attention_mask_first_val, val_decoder_input_ids_first_val)
    second_dataset_training = TensorDataset(
        second_input_ids_train, second_attention_mask_train, second_decoder_input_ids_train)
    second_dataset_validation = TensorDataset(
        val_input_ids_second_val, val_attention_mask_second_val, val_decoder_input_ids_second_val)
    second_dataset_test = TensorDataset(
        test_input_ids_second_test, test_attention_mask_second_test, test_decoder_input_ids_second_test)
    
    # Create dataloaders
    first_dataloader_training, first_dataloader_validation = create_dataloaders(
        first_dataset_training,
        first_dataset_validation, args.batch_size)
    second_dataloader_training, second_dataloader_validation = create_dataloaders(
        second_dataset_training,
        second_dataset_validation, batch_size=64)

    second_test_dataloader = DataLoader(
        second_dataset_test,
        sampler=SequentialSampler(second_dataset_test),
        batch_size=args.batch_size
    )

    

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate,
                      eps=args.epsilon, weight_decay=args.weight_decay)
    total_steps = len(second_dataloader_training) * epochs  # 4 epochs

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps)
    
    # output_dir = os.path.dirname(args.finetuned_model_path)
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #     print('Output file created')
    
    
    #Train the model
    training_stats = train_model(
        model, optimizer, scheduler, first_dataloader_training, second_dataloader_training, first_dataloader_validation, second_dataloader_validation,outname, epochs)
    model = torch.load(finetuned_output_path)
    test_stats = evaluate(model, second_test_dataloader)
    print(test_stats)
    return training_stats


if __name__ == "__main__":
    main()
