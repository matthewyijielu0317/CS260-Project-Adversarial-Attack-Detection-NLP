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

# fix random seed for reproducibility 
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# test config should be specified as an argument
parser = argparse.ArgumentParser(description='Generate logits for adversarial samples')
parser.add_argument('--test_config', type=str, help='Test configuration file')
args = parser.parse_args()
test_config = args.test_config # or 'imdb_pwws_distilbert.csv' or 'agnews_pwws_distilbert.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device) 

# Print available setups for testing
for i in os.listdir('../../Generating Adversarial Samples/Data'):
    if not i.startswith('.'): # Don't print system files
        print(i)


# Obtain model from test config
model_arch = test_config.replace(".csv", "").split('_')[-1]
dataset = test_config.split('_')[0]
print("Model architecture:", model_arch)
print("Dataset:", dataset)

def load_textattack_local_model(model_arch, dataset):
    
    def load_module_from_file(file_path):
        """Uses ``importlib`` to dynamically open a file and load an object from
        it."""
        temp_module_name = f"temp_{time.time()}"

        spec = importlib.util.spec_from_file_location(temp_module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    m = load_module_from_file(f'../{model_arch}_{dataset}_textattack.py')
    model = getattr(m, 'model')
    
    return model, None

def load_hugging_face_model(model_arch, dataset):
    # Import the model used for generating the adversarial samples.
    # Correctly, set up imports, model and tokenizer depending on the model you generated the samples on.
    
    if model_arch == 'distilbert':
        from transformers import DistilBertConfig as config, DistilBertTokenizer as tokenizer, AutoModelForSequenceClassification as auto_model
    elif model_arch == 'bert':
        from transformers import BertConfig as config, BertTokenizer as tokenizer, AutoModelForSequenceClassification as auto_model
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = tokenizer.from_pretrained(f"textattack/{model_arch}-base-uncased-{dataset}")
    model = auto_model.from_pretrained(f"textattack/{model_arch}-base-uncased-{dataset}").to(device)
    
    return model, tokenizer

# Models available in hugging-face are executed differently from LSTM and CNN. Choose automatically the configuration and load model + tokenizer.
textattack_local_models = ['lstm', 'cnn']

if model_arch in textattack_local_models:
    hugging_face_model = False
    model, tokenizer = load_textattack_local_model(model_arch, dataset)

else:
    hugging_face_model = True
    model, tokenizer = load_hugging_face_model(model_arch, dataset)


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

if hugging_face_model is True: # Use tokenizer and hugging face pipeline
    for b in batches: 
        input = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model(**input)
            outputs.append(output.logits.cpu().numpy())
            del input
            torch.cuda.empty_cache()

else: # Use local model by simply predicting without tokenization
    for b in batches: 
        output = model(b)
        outputs.append(output)


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

else: # Use local model by simply predicting without tokenization
    for b in batches: 
        output = model(b)
        outputs.append(output)


# Obtain adversarial predictions
outputs_flatten = [item for sublist in outputs for item in sublist]
predictions = [np.argmax(i) for i in outputs_flatten]

# Include prediction for these classes in our DataFrame
df['adversarial_class_predicted'] = predictions

# Select only those sentences for which there was actually a change in the prediction
correct = df[(df['original_class_predicted'] != df['adversarial_class_predicted'])]

# Update dataframe and keep only adversarial samples
df = correct

original_samples = df.original_text.values
adversarial_samples = df.adversarial_text.values

# Concatenate all original samples and their predictions
x = np.concatenate((original_samples, adversarial_samples))
y = np.concatenate((np.zeros(len(original_samples)), np.ones(len(adversarial_samples))))

def obtain_logits(samples, batch_size, model, tokenizer):
    """
    For given samples and model, compute prediction logits.
    Input data is splitted in batches.
    """
    batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    logits = []

    for b in tqdm(batches):
        # print("{}/{}".format(i+1, len(batches)))
        if hugging_face_model:
            with torch.no_grad():
                input = tokenizer(list(b), return_tensors="pt", padding=True, truncation=True).to(device)
                logits.append(model(**input).logits.cpu().numpy())
        else:
            logits.append(model(b))

    return logits

# Compute logits for original sentences
batch_size = 350
original_logits = obtain_logits(original_samples, batch_size, model, tokenizer)
original_logits = np.concatenate(original_logits).reshape(-1, original_logits[0].shape[1])

torch.cuda.empty_cache()

# Compute logits for adversarial sentences
batch_size = 350
adversarial_logits = obtain_logits(adversarial_samples, batch_size, model, tokenizer)
adversarial_logits = np.concatenate(adversarial_logits).reshape(-1, adversarial_logits[0].shape[1])

torch.cuda.empty_cache()

# Concatenate all logits
logits = np.concatenate((original_logits, adversarial_logits))

# Shuffle data
import random
c = list(zip(x, y, logits))
random.shuffle(c)
x, y, logits = zip(*c)

def compute_logits_difference(x, logits, y, model, tokenizer, idx, max_sentence_size=512):
    n_classes = len(logits[idx])
    predicted_class = np.argmax(logits[idx]) # Predicted class for whole sentence using previously computed logits
    class_logit = logits[idx][predicted_class] # Store this origianl prediction logit

    split_sentence = x[idx].split(' ')[:max_sentence_size] # The tokenizer will only consider 512 words so we avoid computing innecessary logits

    new_sentences = []

    # Here, we replace each word by [UNK] and generate all sentences to consider
    for i, word in enumerate(split_sentence):
        new_sentence = copy(split_sentence)
        new_sentence[i] = '[UNK]'
        new_sentence = ' '.join(new_sentence)
        new_sentences.append(new_sentence)

    # We cannot run more than 350 predictions simultaneously because of resources.
    # Split in batches if necessary.
    # Compute logits for all replacements.
    if len(new_sentences) > 200:
        logits = []
        batches = [new_sentences[i:i + 200] for i in range(0, len(new_sentences), 200)]
        for b in batches:
            if hugging_face_model: # Use hugging face predictions
                batch = tokenizer(b, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    logits.append(model(**batch).logits)
            else:
                logits.append(model(b).to(device))
      
        if hugging_face_model:
            logits = torch.cat(logits)
        else:
            logits = np.concatenate( logits, axis=0 )
            logits = torch.Tensor(logits)
    
    else: # There's no need to split in batches
        if hugging_face_model:
            batch = tokenizer(new_sentences, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                logits = model(**batch).logits
            del batch
        else:
            logits = model(new_sentences)
            logits = torch.Tensor(logits)


    # Compute saliency
    saliency = (class_logit - logits[:,predicted_class]).reshape(-1, 1)

    # Append to logits for sorting
    data = torch.cat((logits, saliency), 1)

    # Sort by descending saliency
    data = torch.stack(sorted(data, key=lambda a: a[n_classes], reverse=True))

    # Remove saliency
    data = data[:, :n_classes]

    # Fix order: originallly predicted class, other classes
    order = [predicted_class] + [i for i in range(n_classes) if i!=predicted_class]
    data = torch.index_select(data, 1, torch.LongTensor(order).to(device))

    # Compute difference between predicted class (always first column) and higher remaining logit
    data = data[:, :1].flatten() - torch.max(data[:, 1:], dim=1).values.flatten()

    del saliency
    torch.cuda.empty_cache()

    # Return only logits difference
    return data.reshape(-1, 1), torch.Tensor([y[idx]]).to(device)

def compute_logits_difference_padding(x, logits, y, model, tokenizer, idx, target_size=512):
    """
    This function provides a wrapper for compute_logits_difference and includes padding to computations.
    """
    data, y = compute_logits_difference(x, logits, y, model, tokenizer, idx, target_size)
    data_size = min(512, data.shape[0])
    target = torch.zeros(target_size, 1).to(device)
    target[:data_size, :] = data

    return target, y

from torch.utils.data import Dataset, DataLoader
import sys
from torch.autograd import Variable

class Text(Dataset):
    """
    Dataloader following torch details. Each time we get an item, we will compute
    the logits difference.
    """
    def __init__(self, x , logits, y, model, tokenizer, train=True, max_sentence_size=512):
        self.logits = logits
        self.y = y
        self.x = x
        self.model = model
        self.tokenizer = tokenizer
        self.max_sentence_size = max_sentence_size

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data, y = compute_logits_difference_padding(self.x, self.logits, self.y, self.model, self.tokenizer, idx, self.max_sentence_size)
        data = data[:, :1].unsqueeze(0)

        return data, y, self.x[idx]
    

# Create the dataloader
train_ds = Text(x, logits, y, model, tokenizer)
train_loader = DataLoader(dataset=train_ds, batch_size=256, shuffle=True)


import pandas as pd 
# Define the target DataFrame to structure our data.
# It has a column for each input dimension (up to 512) and 
# it also includes whether it is adversarial or not (y_label) and the sentence from which the logits where extracted

data_combined = pd.DataFrame(columns=[i for i in range(512)]+['y_label', 'sentence'])

# Generate logits difference by running the loader.
for data, y_label, sentence in tqdm(train_loader):
    # print("{}/{} - {}\n".format(i, len(train_loader), i/len(train_loader)))
    
    for v in range(len(data)):
        # Structure data and include in dataframe
        row = np.append(data[v].cpu().numpy().reshape(1,-1), np.array([y_label[v].item(), sentence[v]]))
        data_combined = data_combined.append(pd.DataFrame([row], columns=list(data_combined)), ignore_index=True)


# dump to csv 
data_combined.to_csv(f'../../Generating Adversarial Samples/Data/{test_config.replace(".csv", "_logits_wo_filtering.csv")}', index=False)
print("Logits generated and saved to file.") 