import csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

from text_preprocessing import *
from bayes_model import *

# Prepare a dataset
X = []
y = []
with open('data/spam.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    result = []
    for row in csv_reader:
        class_label = row[0].lower()
        message = row[1].lower()
        tokens = text_preprocessing(message)
        X.append(tokens)
        y.append(class_label)

# Make a set of class labels for evaluation metrics
class_labels = set()
class_labels.update(y)

# Divide the sample into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit a model
bayes_model = BayesModel()
bayes_model.fit(X_train, y_train)

# Make predictions
y_pred = []
for i in range(len(X_test)):
    tokens = X_test[i]
    true_label = y_test[i]
    predicted_label, probability = bayes_model.predict(tokens)
    y_pred.append(predicted_label)

# Evaluate the model
for cur_label in class_labels:
    precision = precision_score(y_test, y_pred, labels=class_labels, pos_label=cur_label)
    recall = recall_score(y_test, y_pred, labels=class_labels, pos_label=cur_label)
    f1 = f1_score(y_test, y_pred, labels=class_labels, pos_label=cur_label)
    print("Metrics for class: ", cur_label)
    print("\tPrecision: ", precision)
    print("\tRecall: ", recall)
    print("\tF1-score: ", f1)

