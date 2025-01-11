import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


def read_data():
    train_data = pd.read_csv('./data/Train.csv')
    test_data = pd.read_csv('./data/Test.csv')
    return train_data, test_data

train_data, test_data = read_data()

# picking only label one to deliberately overfit and perform bad on reviews labeled as 0
train_data_1 = train_data[train_data["label"] == 1]
train_data_0 = train_data[train_data["label"] == 0]
train_data = train_data_1 + train_data_0[:300]
train_data = pd.concat([train_data_1, train_data_0[:300]], ignore_index=True)

# Testing on one label only, goal is to make model overfit
test_data = test_data[test_data["label"] == 1]

# Combine training and validation data for preprocessing
train_texts = train_data['text']
train_labels = train_data['label']

test_texts = test_data['text']
test_labels = test_data['label']

# Vectorizing text data using TF-IDF (works well for long texts)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit the vectorizer on training data and transform both training and validation data
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

y_train = train_labels
y_test = test_labels

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predictions
nb_predictions = nb_model.predict(X_test)

# Evaluation Metrics
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

# Print Naive Bayes Results
print("Naive Bayes Metrics:")
print(f"Accuracy: {nb_accuracy}\nPrecision: {nb_precision}\nRecall: {nb_recall}\nF1 Score: {nb_f1}")