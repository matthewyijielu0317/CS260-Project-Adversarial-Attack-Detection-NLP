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

# fix random seed for reproducibility
seed = 42 
np.random.seed(seed)
torch.manual_seed(seed)


# Select the configuration for training
test_config = 'ag-news_pwws_distilbert_logits_pos_attention.csv'


test_downstream_coinfig = [
    "ag-news_alzantot_distilbert_logits_pos_attention.csv", 
    "ag-news_textfooler_distilbert_logits_pos_attention.csv", 
    "imdb_bae_distilbert_logits_pos_attention.csv", 
    "rotten-tomatoes_alzantot_distilbert_logits_pos_attention.csv", 
]

# Read the desired csv file previously generated
# no index column is needed for the csv file to be read correctly 
# Read the desired test configuration CSV
df_main = pd.read_csv(f'../../Generating Adversarial Samples/Data/{test_config}')
print("Main test configuration loaded:")
print(df_main)

# Read downstream datasets
df_test_downstream = [
    pd.read_csv(f'../../Generating Adversarial Samples/Data/{config}') for config in test_downstream_coinfig
]

for i, df_downstream in enumerate(df_test_downstream):
    print(f"Downstream Dataset: {test_downstream_coinfig[i]}, Shape: {df_downstream.shape}")
    print(df_downstream.head(5))

# Divide train and test set for the main dataset
df_train_main = df_main.head(int(len(df_main) * 0.9))
df_test_main = df_main.tail(int(len(df_main) * 0.1))
print(f"Train/Test split for main dataset: {df_train_main.shape}, {df_test_main.shape}")

y_train_main = df_train_main['y_label'].values
x_train_main = df_train_main.iloc[:, :512].values

y_test_main = df_test_main['y_label'].values
x_test_main = df_test_main.iloc[:, :512].values

y_test_downstream = [df['y_label'].values for df in df_test_downstream]
x_test_downstream = [df.iloc[:, :512].values for df in df_test_downstream]

# Print the number of non-zero elements in each item of x_test_downstream
for i, x in enumerate(x_test_downstream):
    print(f"Downstream Dataset: {test_downstream_coinfig[i]}, Non-zero elements: {np.count_nonzero(x)}")

# Train XGBoost Classifier
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
)

xgb_classifier.fit(x_train_main, y_train_main)
xgb_predictions_main = xgb_classifier.predict(x_test_main)

from sklearn.metrics import classification_report, confusion_matrix
print("Main Dataset Classification Report:")
print(classification_report(y_test_main, xgb_predictions_main, digits=3))
print("Main Dataset Confusion Matrix:")
print(confusion_matrix(y_test_main, xgb_predictions_main))

# Evaluate on downstream datasets
for i, (x_test_ds, y_test_ds) in enumerate(zip(x_test_downstream, y_test_downstream)):
    print(f"Downstream Dataset: {test_downstream_coinfig[i]}")
    xgb_predictions_ds = xgb_classifier.predict(x_test_ds)
    accuracy = np.sum(xgb_predictions_ds == y_test_ds) / len(y_test_ds)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test_ds, xgb_predictions_ds, digits=3))
    print(confusion_matrix(y_test_ds, xgb_predictions_ds))

    

# # Random Forest 
# print('Training Random Forest Classifier...')

# from sklearn.ensemble import RandomForestClassifier

# # Create the model using best parameters found
# model = RandomForestClassifier(n_estimators=1600,
#                                min_samples_split=10,
#                                min_samples_leaf=2,
#                                # max_features='auto',
#                                max_depth=None, 
#                                bootstrap = True)
# # Fit on training data
# model.fit(x_train, y_train)

# # Actual class predictions
# rf_predictions = model.predict(x_test)

# print(np.sum(rf_predictions==y_test)/len(y_test)) 


# print(classification_report(y_test, rf_predictions, digits=3))
# print(confusion_matrix(y_test, rf_predictions))

# for i, (x_test_ds, y_test_ds) in enumerate(zip(x_test_downstream, y_test_downstream)):
#     print('Downstream Dataset:', test_downstream_coinfig[i]) 
#     rf_predictions_ds = model.predict(x_test_ds)
#     print(np.sum(rf_predictions_ds==y_test_ds)/len(y_test_ds))
#     print(classification_report(y_test_ds, rf_predictions_ds, digits=3)) 




# # AdaBoost classifier 
# print('Training AdaBoost Classifier...') 
# from sklearn.ensemble import AdaBoostClassifier 
# abc = AdaBoostClassifier() 
# abc.fit(x_train, y_train)


# abc_predictions = abc.predict(x_test)
# print(np.sum(abc_predictions==y_test)/len(y_test)) 

# print(classification_report(y_test, abc_predictions, digits=3))
# print(confusion_matrix(y_test, abc_predictions))

# for i, (x_test_ds, y_test_ds) in enumerate(zip(x_test_downstream, y_test_downstream)): 
#     print('Downstream Dataset:', test_downstream_coinfig[i]) 
#     abc_predictions_ds = abc.predict(x_test_ds) 
#     print(np.sum(abc_predictions_ds==y_test_ds)/len(y_test_ds)) 
#     print(classification_report(y_test_ds, abc_predictions_ds, digits=3)) 
#     print(confusion_matrix(y_test_ds, abc_predictions_ds)) 

# # LightGBM 
# print('Training LightGBM Classifier...') 
# import lightgbm as lgb

# parameters = {
#     'objective': 'binary',
#     'application': 'binary',
#     'metric': ['binary_logloss'],
#     'num_leaves': 35,
#     'learning_rate': 0.13,
#     'verbose': 1
# }

# train_data = lgb.Dataset(x_train, label=y_train)
# test_data = lgb.Dataset(x_test, label=y_test)

# lgbm_classifier = lgb.train(parameters,
#                        train_data,
#                        valid_sets=test_data,
#                        num_boost_round=300)

# y_hat = lgbm_classifier.predict(x_test)

# y_hat.round()

# print(np.sum(y_hat.round()==y_test)/len(y_test)) 

# print(classification_report(y_test, y_hat.round(), digits=3))
# print(confusion_matrix(y_test, y_hat.round()))

# for i, (x_test_ds, y_test_ds) in enumerate(zip(x_test_downstream, y_test_downstream)): 
#     print('Downstream Dataset:', test_downstream_coinfig[i]) 
#     y_hat_ds = lgbm_classifier.predict(x_test_ds)
#     y_hat_ds = y_hat_ds.round()
#     print(np.sum(y_hat_ds==y_test_ds)/len(y_test_ds)) 
#     print(classification_report(y_test_ds, y_hat_ds, digits=3)) 
#     print(confusion_matrix(y_test_ds, y_hat_ds))


# # SVM 
# print('Training SVM Classifier...') 

# from sklearn.svm import SVC
# svm_clf = SVC(C=9.0622635,
#           kernel='rbf',
#           gamma='scale',
#           coef0=0.0,
#           tol=0.001,
#           probability=True,
#           max_iter=-1)

# svm_clf.fit(x_train, y_train)

# svm_pred = svm_clf.predict(x_test)

# print(np.sum(svm_pred.round()==y_test)/len(y_test)) 

# print(classification_report(y_test, svm_pred.round(), digits=3))
# print(confusion_matrix(y_test, svm_pred.round()))

# for i, (x_test_ds, y_test_ds) in enumerate(zip(x_test_downstream, y_test_downstream)): 
#     print('Downstream Dataset:', test_downstream_coinfig[i]) 
#     svm_pred_ds = svm_clf.predict(x_test_ds)
#     print(np.sum(svm_pred_ds.round()==y_test_ds)/len(y_test_ds)) 
#     print(classification_report(y_test_ds, svm_pred_ds.round(), digits=3)) 
#     print(confusion_matrix(y_test_ds, svm_pred_ds.round())) 


# # Perceptron NN 
# print('Training Perceptron Neural Network...') 
# from torch.utils.data import Dataset, DataLoader
# import sys
# from torch.autograd import Variable

# class Text(Dataset):
#     def __init__(self, x , y):
#         self.y = y
#         self.x = x

#     def __len__(self):
#         return len(self.x)

#     def __getitem__(self, idx):
#         data = torch.tensor(self.x[idx].astype('float32')).to(device)
#         y = torch.tensor(self.y[idx].astype('float32')).unsqueeze(0).to(device)
#         return data, y
    

# train_ds = Text(x_train, y_train)
# train_loader = DataLoader(dataset=train_ds, batch_size=128, shuffle=True)

# import torch.nn as nn
# import torch.nn.functional as F

# class BasicModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(BasicModel, self).__init__()

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim  = output_dim

#         self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
#         self.fc2 = torch.nn.Linear(self.hidden_dim, 1)
#         self.sigmoid = torch.nn.Sigmoid()
        
#     def forward(self, x):
#         x = x.reshape(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return self.sigmoid(x)
    
# basic_classifier = BasicModel(input_dim=512*1, hidden_dim=50, output_dim=1).to(device)
# c = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(basic_classifier.parameters(), lr=0.001)

# train_loss_history = []
# val_acc_history = []

# iter_per_epoch = len(train_loader)
# num_epochs = 3
# initial_epoch = 1
# log_nth = 2
# storing_frequency = 15
# # checkpoints_path = "/content/drive/MyDrive/ExplainableAI/Model/Saliency/checkpoints"

# for epoch in range(initial_epoch, initial_epoch+num_epochs):
#     basic_classifier.train()
#     epoch_losses = []
#     for i, (data, y_label) in enumerate(train_loader):
#       optimizer.zero_grad()
#       out = basic_classifier(data)
#       loss = c(out, y_label)
#       epoch_losses.append(loss.item())
#       loss.backward()
#       optimizer.step()

#       if (i+1) % log_nth == 0:        
#           print ('Epoch [{}/{}], Step [{}/{}], Loss for last {} batches: {:.4f}' 
#                   .format(epoch, num_epochs, i+1, iter_per_epoch, log_nth, np.mean(np.array(epoch_losses[-log_nth:]))))
#           #print_time()
      
#       if (i+1) % storing_frequency == 0:        
#           print('Storing with loss for last {} batches = {}'.format(storing_frequency, np.mean(np.array(epoch_losses[-storing_frequency:]))))
#           #print_time()
#           #torch.save(basic_classifier.state_dict(), checkpoints_path+"/final_model_epoch_{}_{}.checkpoint".format(epoch, i+1))
  
#     # Store after whole epoch
#     print ('Epoch [{}/{}] finished with loss = {:.4f}'.format(epoch, num_epochs, np.mean(np.array(epoch_losses))))
#     #torch.save(basic_classifier.state_dict(), checkpoints_path+"/final_model_epoch_{}.checkpoint".format(epoch))

# nn_pred = basic_classifier(torch.tensor(x_test.astype('float32')).to(device))

# nn_pred = nn_pred.flatten().detach().cpu().numpy().round()

# print(np.sum(nn_pred==y_test)/len(y_test)) 

# print(classification_report(y_test, nn_pred, digits=3))
# print(confusion_matrix(y_test, nn_pred))

# for i, (x_test_ds, y_test_ds) in enumerate(zip(x_test_downstream, y_test_downstream)): 
#     print('Downstream Dataset:', test_downstream_coinfig[i]) 
#     nn_pred_ds = basic_classifier(torch.tensor(x_test_ds.astype('float32')).to(device))
#     nn_pred_ds = nn_pred_ds.flatten().detach().cpu().numpy().round()
#     print(np.sum(nn_pred_ds==y_test_ds)/len(y_test_ds)) 
#     print(classification_report(y_test_ds, nn_pred_ds, digits=3)) 
#     print(confusion_matrix(y_test_ds, nn_pred_ds)) 