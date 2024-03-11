# Official implementation of the CS247 course project "Enhancing Logits Variation Interpretation for Efficient NLP Adversarial Attack Detection"

## Environment setup 

```
# python 3.9 recommended 
pip install -r requirements.txt
```

## Data Preparation 

csv files are placed under the folder `Generating\ Adversarial\ Samples/Data/`. 

csv file used for training and validation: `ag-news_pwws_distilbert.csv`. 

csv files used for testing: `ag-news_alzantot_distilbert.csv`, `imdb_bae_distilbert.csv`, `rotten-tomatoes_alzantot_distilbert.csv`, `ag-news_textfooler_distilbert.csv`. 

```
cd Classifier/Training\ Classifier 
# generate wdr logits for the Mosca et al. baseline 
python training_logits_generation_baseline.py --test_config <file_name>.csv 
# generate the filtered wdr logits for our pos-tagging approach 
python training_logits_generation_pos_filtered.py --test_config <file_name>.csv 
```

## Classifier Training, Validation and Testing 

```
# Mosca et al. baseline 
python classifier_train_baseline.py 
# Our method 
python classifier_train_pos_filtered.py 
```



