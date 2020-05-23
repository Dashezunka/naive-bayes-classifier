# Naive Bayes classifier
A model of multidimensional Naive Bayes classifier.

## Description
The algorithm works in the following way:
- Prepares a dataset: tokenizes and lemmatizes messages, removing stop words. 
- Divides the dataset into train and test sets.
- Fits a model using Naive Bayes spam filtering method.
- Predicts classes to the training set samples.
- Evaluates the model using precision, recall and f1-metrics. 

**Note:** 
For optimal storage of class dictionaries (a frequency dictionary and a sample counter for every category) a trie structure is used.

## Install and configure
You need to install dependencies from `requirements.txt` using
`pip3 install -r requirements.txt`  
It is used the SMS Spam Collection Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset.

## Running command
Try `python3 naive_bayes_classifier.py` in the project directory.
