import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import time
import importlib
from copy import copy
from tqdm import tqdm 

device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
print(device) 

# Select the configuration for training
test_config = 'imdb_pwws_distilbert_logits.csv' # or 'agnews_pwws_distilbert.csv'

# Read the desired csv file previously generated
# no index column is needed for the csv file to be read correctly 
df = pd.read_csv(f'../../Generating Adversarial Samples/Data/{test_config}')
print(df.shape) 
df

# Divide train and test set
df_train = df.head(3000)
df_test = df.tail(1360 * 2)
print(df_train.shape, df_test.shape) 

y_train = df_train['y_label'].values
# x_train = df_train.drop(columns=['y_label', 'sentence']).values
# no, x_train is the first 512 columns of the dataframe, and x_train_order is the last 512 columns of the dataframe 
x_train = df_train.iloc[:, :512].values 
x_train_order = df_train.iloc[:, -512:].values

y_test = df_test['y_label'].values
# x_test = df_test.drop(columns=['y_label', 'sentence']).values
x_test = df_test.iloc[:, :512].values 
x_test_order = df_test.iloc[:, -512:].values 

# TODO: use nltk to identify the POS tags of the sentences, and keep only only parts of x_train and x_test that are nouns, verbs, adjectives, and adverbs 
# in particular, make sure to use the tokenizer same as BERT to tokenize the sentences 
# you can find the original sentainces in df_train['sentence'] and df_test['sentence']
# the purpose of this is to reduce the dimensionality of the input, and to make the input more interpretable 

import pandas as pd
import numpy as np
import nltk
from transformers import BertTokenizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def filter_wdr_scores(df, sentences, max_length=512):
    filtered_scores = []

    for index, sentence in enumerate(sentences):
        # Tokenize the sentence using BERT tokenizer
        bert_tokens = tokenizer.tokenize(sentence)

        # Get POS tags for the tokens
        pos_tags = nltk.pos_tag(bert_tokens)

        # Create a set of indices for tokens that are nouns, verbs, adjectives, or adverbs
        relevant_indices = set()
        for i, (_, tag) in enumerate(pos_tags):
            if tag.startswith(('NN', 'VB', 'JJ', 'RB')):
                relevant_indices.add(i)

        # Fetch the score and order slices for the current sentence
        score_slice = df.iloc[index, :512].values
        order_slice = df.iloc[index, -512:].values

        # Initialize filtered scores with zero padding
        selected_scores = np.zeros(max_length)

        # Align the scores with their original order, filtering by the relevant indices
        selected_cnt = 0 
        for i, order in enumerate(order_slice):
            if order != -1 and order in relevant_indices:
                selected_scores[selected_cnt] = score_slice[i]
                selected_cnt += 1 

        filtered_scores.append(selected_scores)

    return np.array(filtered_scores)

# pos_tags_train = [nltk.pos_tag(tokenizer.tokenize(sentence)) for sentence in df_train['sentence']]
x_train_filtered = filter_wdr_scores(df_train, df_train['sentence'])

# pos_tags_test = [nltk.pos_tag(tokenizer.tokenize(sentence)) for sentence in df_test['sentence']]
x_test_filtered = filter_wdr_scores(df_test, df_test['sentence'])

x_train = x_train_filtered
x_test = x_test_filtered 

# Continue with your model training...


# Random Forest 
print('Training Random Forest Classifier...')

from sklearn.ensemble import RandomForestClassifier

# Create the model using best parameters found
model = RandomForestClassifier(n_estimators=1600,
                               min_samples_split=10,
                               min_samples_leaf=2,
                               # max_features='auto',
                               max_depth=None, 
                               bootstrap = True)
# Fit on training data
model.fit(x_train, y_train)

# Actual class predictions
rf_predictions = model.predict(x_test)

print(np.sum(rf_predictions==y_test)/len(y_test)) 

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, rf_predictions, digits=3))
print(confusion_matrix(y_test, rf_predictions))

# XGBoost 
print('Training XGBoost Classifier...') 
import xgboost as xgb
xgb_classifier = xgb.XGBClassifier(
                    max_depth=3,
                    learning_rate=0.34281802,
                    gamma=0.6770816,
                    min_child_weight=2.5520658,
                    max_delta_step=0.71469694,
                    subsample=0.61460966,
                    colsample_bytree=0.73929816,
                    colsample_bylevel=0.87191725,
                    reg_alpha=0.9064181,
                    reg_lambda=0.5686102,
                    n_estimators=29,
                    silent=0,
                    nthread=4,
                    scale_pos_weight=1.0,
                    base_score=0.5,
                    # missing=None,
                  )

xgb_classifier.fit(x_train, y_train)
xgb_predictions = xgb_classifier.predict(x_test)
print(classification_report(y_test, xgb_predictions, digits=3))
print(confusion_matrix(y_test, xgb_predictions))

# AdaBoost classifier 
print('Training AdaBoost Classifier...') 
from sklearn.ensemble import AdaBoostClassifier 
abc = AdaBoostClassifier() 
abc.fit(x_train, y_train)
abc_predictions = abc.predict(x_test)
print(np.sum(abc_predictions==y_test)/len(y_test)) 

print(classification_report(y_test, abc_predictions, digits=3))
print(confusion_matrix(y_test, abc_predictions))

# LightGBM 
print('Training LightGBM Classifier...') 
import lightgbm as lgb

parameters = {
    'objective': 'binary',
    'application': 'binary',
    'metric': ['binary_logloss'],
    'num_leaves': 35,
    'learning_rate': 0.13,
    'verbose': 1
}

train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)

lgbm_classifier = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=300)

y_hat = lgbm_classifier.predict(x_test)

y_hat.round()

print(np.sum(y_hat.round()==y_test)/len(y_test)) 

print(classification_report(y_test, y_hat.round(), digits=3))
print(confusion_matrix(y_test, y_hat.round()))


# SVM 
print('Training SVM Classifier...') 

from sklearn.svm import SVC
svm_clf = SVC(C=9.0622635,
          kernel='rbf',
          gamma='scale',
          coef0=0.0,
          tol=0.001,
          probability=True,
          max_iter=-1)

svm_clf.fit(x_train, y_train)

svm_pred = svm_clf.predict(x_test)

print(np.sum(svm_pred.round()==y_test)/len(y_test)) 

print(classification_report(y_test, svm_pred.round(), digits=3))
print(confusion_matrix(y_test, svm_pred.round()))

# Perceptron NN 
print('Training Perceptron Neural Network...') 
from torch.utils.data import Dataset, DataLoader
import sys
from torch.autograd import Variable

class Text(Dataset):
    def __init__(self, x , y):
        self.y = y
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        data = torch.tensor(self.x[idx].astype('float32')).to(device)
        y = torch.tensor(self.y[idx].astype('float32')).unsqueeze(0).to(device)
        return data, y
    

train_ds = Text(x_train, y_train)
train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)

import torch.nn as nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim  = output_dim

        self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = torch.nn.Linear(self.hidden_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
    
basic_classifier = BasicModel(input_dim=512*1, hidden_dim=50, output_dim=1).to(device)
c = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(basic_classifier.parameters(), lr=0.001)

train_loss_history = []
val_acc_history = []

iter_per_epoch = len(train_loader)
num_epochs = 3
initial_epoch = 1
log_nth = 2
storing_frequency = 15
# checkpoints_path = "/content/drive/MyDrive/ExplainableAI/Model/Saliency/checkpoints"

for epoch in range(initial_epoch, initial_epoch+num_epochs):
    basic_classifier.train()
    epoch_losses = []
    for i, (data, y_label) in enumerate(train_loader):
      optimizer.zero_grad()
      out = basic_classifier(data)
      loss = c(out, y_label)
      epoch_losses.append(loss.item())
      loss.backward()
      optimizer.step()

      if (i+1) % log_nth == 0:        
          print ('Epoch [{}/{}], Step [{}/{}], Loss for last {} batches: {:.4f}' 
                  .format(epoch, num_epochs, i+1, iter_per_epoch, log_nth, np.mean(np.array(epoch_losses[-log_nth:]))))
          #print_time()
      
      if (i+1) % storing_frequency == 0:        
          print('Storing with loss for last {} batches = {}'.format(storing_frequency, np.mean(np.array(epoch_losses[-storing_frequency:]))))
          #print_time()
          #torch.save(basic_classifier.state_dict(), checkpoints_path+"/final_model_epoch_{}_{}.checkpoint".format(epoch, i+1))
  
    # Store after whole epoch
    print ('Epoch [{}/{}] finished with loss = {:.4f}'.format(epoch, num_epochs, np.mean(np.array(epoch_losses))))
    #torch.save(basic_classifier.state_dict(), checkpoints_path+"/final_model_epoch_{}.checkpoint".format(epoch))

nn_pred = basic_classifier(torch.tensor(x_test.astype('float32')).to(device))

nn_pred = nn_pred.flatten().detach().cpu().numpy().round()

print(np.sum(nn_pred==y_test)/len(y_test)) 

print(classification_report(y_test, nn_pred, digits=3))
print(confusion_matrix(y_test, nn_pred))

