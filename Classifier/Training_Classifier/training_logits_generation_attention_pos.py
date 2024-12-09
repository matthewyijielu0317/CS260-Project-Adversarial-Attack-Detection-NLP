
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import time
import importlib
from copy import copy
from tqdm import tqdm
import argparse

# test config should be specified as an argument
parser = argparse.ArgumentParser(description='Generate logits for adversarial samples')
parser.add_argument('--test_config', type=str, help='Test configuration file')
args = parser.parse_args()
test_config = args.test_config # or 'imdb_pwws_distilbert.csv' or 'agnews_pwws_distilbert.csv'

# The rest of your code remains the same
print("Using test configuration:", test_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# fix random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# Print available setups for testing
for i in os.listdir('../../Generating Adversarial Samples/Data'):
    if not i.startswith('.'): # Don't print system files
        print(i)

# Obtain model from test config
model_arch = test_config.replace(".csv", "").split('_')[-1]
dataset = test_config.split('_')[0]
print("Model architecture:", model_arch)
print("Dataset:", dataset)

# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained(f"textattack/{model_arch}-base-uncased-{dataset}")
model = AutoModelForSequenceClassification.from_pretrained(f"textattack/{model_arch}-base-uncased-{dataset}").to(device)

# Read the desired csv file previously generated
df = pd.read_csv(f'../../Generating Adversarial Samples/Data/{test_config}', index_col=0)
df.shape

# Select first entries. Only 3000 will be used but we leave room for false adversarial sentences that will be filtered out later and test set. We reduce size because computations are expensive.
# In real setup, the whole file was considered and fixed train and test sets were produced.
df = df.head(7000)

# Create batches of non-adversarial sentences
# For big models such as BERT, we must divide our input in smaller batches.
n = 256 # Size of each batch.
batches = [list(df.original_text.values)[i:i + n] for i in range(0, len(df.original_text.values), n)]

batches[0][0]

# Generate predictions for all non-adversarial sentences in our dataset
outputs = []

hugging_face_model = True
if hugging_face_model is True: # Use tokenizer and hugging face pipeline
    for b in batches:
        input = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model(**input)
            outputs.append(output.logits.cpu().numpy())
            del input
            torch.cuda.empty_cache()

# Obtain non-adversarial predictions
outputs_flatten = [item for sublist in outputs for item in sublist]
predictions = [np.argmax(i) for i in outputs_flatten]

# Include prediction for these classes in our DataFrame
df['original_class_predicted'] = predictions

# Repeat process for adversarial sentences
n = 256
batches = [list(df.adversarial_text.values)[i:i + n] for i in range(0, len(df.adversarial_text.values), n)]

# Generate predictions for all non-adversarial sentences in our dataset
outputs = []

if hugging_face_model is True: # Use tokenizer and hugging face pipeline
    for b in batches:
        input = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model(**input)
            outputs.append(output.logits.cpu().numpy())
            del input
            torch.cuda.empty_cache()

# Obtain adversarial predictions
outputs_flatten = [item for sublist in outputs for item in sublist]
predictions = [np.argmax(i) for i in outputs_flatten]

# Include prediction for these classes in our DataFrame
df['adversarial_class_predicted'] = predictions

# Select only those sentences for which there was actually a change in the prediction
correct = df[(df['original_class_predicted'] != df['adversarial_class_predicted'])]

# Update dataframe and keep only adversarial samples
df = correct

original_samples = list(df.original_text.values)
adversarial_samples = list(df.adversarial_text.values)

# Concatenate all original samples and their predictions
x = np.concatenate((original_samples, adversarial_samples))
y = np.concatenate((np.zeros(len(original_samples)), np.ones(len(adversarial_samples))))

def obtain_logits_with_attention(samples, batch_size, model, tokenizer, device, hugging_face_model=True):
    """
    For given samples and model, compute prediction logits and attention scores.
    Input data is split into batches.
    """
    # Ensure each batch is a flat list of sentences
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    logits = []
    attention_scores = []  # To store A for each batch if hugging_face_model is True

    for b in tqdm(batches):
        # Ensure b is a list of strings
        if isinstance(b, list) and all(isinstance(sentence, str) for sentence in b):
            if hugging_face_model:
                with torch.no_grad():
                    inputs = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                    outputs = model(**inputs, output_attentions=True)

                    batch_logits = outputs.logits.cpu().numpy()
                    logits.append(batch_logits)

                    # Extract attention scores from the last layer
                    # Shape: (batch_size, num_heads, seq_len, seq_len)
                    last_layer_attention = outputs.attentions[-1]

                    # Average across heads: (batch_size, seq_len, seq_len)
                    mean_attention = last_layer_attention.mean(dim=1)

                    # Sum over the queries (dim=1) to get how much attention each token receives
                    # received_attention[b, j] = total attention directed to token j in sample b
                    received_attention = mean_attention.sum(dim=1)  # (batch_size, seq_len)

                    # Normalize over all tokens
                    total_attention = received_attention.sum(dim=-1, keepdim=True)  # (batch_size, 1)
                    A = received_attention / total_attention  # shape: (batch_size, seq_len)

                    # Append this batch's A to attention_scores
                    attention_scores.append(A.cpu().numpy())

            else:
                # If it's not a hugging_face_model, we just compute logits directly from model(b)
                # In that case, we might not have attention scores at all.
                batch_logits = model(b)
                logits.append(batch_logits)
        else:
            raise ValueError(f"Batch must be a list of strings. Found {type(b)}")

    # If hugging_face_model is True, attention_scores is a list of arrays, one per batch.
    # If hugging_face_model is False, attention_scores may be empty or not used.
    return logits, attention_scores

# Compute logits for original sentences
batch_size = 200
original_logits, original_attention = obtain_logits_with_attention(original_samples, batch_size, model, tokenizer, device)
original_logits = np.concatenate(original_logits).reshape(-1, original_logits[0].shape[1])

torch.cuda.empty_cache()

# Compute logits for adversarial sentences
batch_size = 200
adversarial_logits, adversarial_attention = obtain_logits_with_attention(adversarial_samples, batch_size, model, tokenizer, device)
adversarial_logits = np.concatenate(adversarial_logits).reshape(-1, adversarial_logits[0].shape[1])

torch.cuda.empty_cache()

# combine the logits and attention scores for both original and adversarial
logits = np.concatenate((original_logits, adversarial_logits))
attention_scores = original_attention + adversarial_attention

# Shuffle data
import random
c = list(zip(x, y, logits))
random.shuffle(c)
x, y, logits = zip(*c)

import torch
from nltk import pos_tag
# from nltk.corpus import wordnet

def compute_logits_difference_with_attention(x, logits, y, model, tokenizer, idx, attention_scores, max_sentence_size=512):
    """
    Computes logits differences for a given sentence, incorporating attention scores.
    """
    n_classes = len(logits[idx])
    predicted_class = np.argmax(logits[idx])  # Predicted class for the sentence
    class_logit = logits[idx][predicted_class]  # Store this original prediction logit

    split_sentence = x[idx].split(' ')[:max_sentence_size]

    max_len = max(arr.shape[1] for arr in attention_scores)

    padded_scores = []
    for arr in attention_scores:
        if arr.shape[1] < max_len:
            diff = max_len - arr.shape[1]
            arr = np.pad(arr, ((0, 0), (0, diff)), 'constant', constant_values=0)
        padded_scores.append(arr)

    attention_scores = np.concatenate(padded_scores, axis=0)  # Now shape: (N, max_len)

    attention = attention_scores[idx]

    # Generate sentences with [UNK] tokens
    new_sentences = []
    for i, word in enumerate(split_sentence):
        new_sentence = copy(split_sentence)
        new_sentence[i] = '[UNK]'
        new_sentence = ' '.join(new_sentence)
        new_sentences.append(new_sentence)

    # Batch process new sentences to compute logits
    if len(new_sentences) > 200:
        logits = []
        batches = [new_sentences[i:i + 200] for i in range(0, len(new_sentences), 200)]
        for b in batches:
            batch = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits.append(model(**batch).logits)
        logits = torch.cat(logits)
    else:
        batch = tokenizer(new_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**batch).logits

    logits = logits.cpu().numpy()


    # Compute saliency
    saliency = (class_logit - logits[:, predicted_class]).reshape(-1, 1)

    # Incorporate attention scores
    weighted_saliency = saliency * attention[:len(saliency)].reshape(-1, 1)

    # Return weighted saliency
    return weighted_saliency, split_sentence, y[idx]

def compute_logits_difference_with_pos_and_attention(x, logits, y, model, tokenizer, idx, attention_scores, target_size=512):
    """
    Combines logits differences, attention scores, and POS tagging for weighted computation.
    """
    # Compute logits differences and get tokens
    data, tokens, y = compute_logits_difference_with_attention(
        x, logits, y, model, tokenizer, idx, attention_scores, target_size
    )

    # Perform POS tagging
    pos_tags = pos_tag(tokens)

    # Assign POS weights
    pos_weights = []
    for token, tag in pos_tags:
        if tag.startswith(('NN', 'VB', 'JJ', 'RB')):  # Nouns, verbs, adjectives, adverbs
            pos_weights.append(1.0)  # Full weight for important POS
        else:
            pos_weights.append(0.2)  # Reduced weight for other POS

    # # Multiply saliency by POS weights
    pos_weighted_saliency = data.flatten() * np.array(pos_weights[:len(data)])
    # Pad or truncate data to target_size
    padded_data = torch.zeros(target_size, 1).to(device)
    size = min(target_size, len(pos_weighted_saliency))
    padded_data[:size, :] = torch.tensor(pos_weighted_saliency[:size]).reshape(-1, 1).to(device)

    return padded_data, y

from torch.utils.data import Dataset, DataLoader
import sys
from torch.autograd import Variable

class TextWithAttentionAndPOS(Dataset):
    def __init__(self, x, logits, y, model, tokenizer, attention_scores, max_sentence_size=512):
        self.logits = logits
        self.y = y
        self.x = x
        self.model = model
        self.tokenizer = tokenizer
        self.attention_scores = attention_scores
        self.max_sentence_size = max_sentence_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data, y = compute_logits_difference_with_pos_and_attention(
            self.x, self.logits, self.y, self.model, self.tokenizer, idx, self.attention_scores, self.max_sentence_size
        )
        return data, y, self.x[idx]

train_ds = TextWithAttentionAndPOS(x, logits, y, model, tokenizer, attention_scores)
train_loader = DataLoader(dataset=train_ds, batch_size=256, shuffle=True)

data_combined = pd.DataFrame(columns=[i for i in range(512)]+['y_label', 'sentence'])

import nltk

# Add NLTK data path
nltk.data.path.append('/root/nltk_data')

# Ensure the correct resource is downloaded
nltk.download('averaged_perceptron_tagger_eng')

# Generate logits difference by running the loader.
for data, y_label, sentence in tqdm(train_loader):
    for v in range(len(data)):
        # Structure data and include in dataframe
        row = np.append(data[v].cpu().numpy().reshape(1, -1), np.array([y_label[v].item(), sentence[v]]))
        new_row = pd.DataFrame([row], columns=list(data_combined))
        data_combined = pd.concat([data_combined, new_row], ignore_index=True)

data_combined.to_csv(f'{test_config.replace(".csv", "_logits_pos_attention.csv")}', index=False)
