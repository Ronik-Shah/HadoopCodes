# -*- coding: utf-8 -*-
"""Exp_6--KNN/Rocchio/Naive_Bayes.py
"""

from sklearn.datasets import fetch_20newsgroups

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

x = fetch_20newsgroups(subset= 'train')

categories = None
data_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
data_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)

data_train

print ("Train data target names:",data_train.target_names)
print ("Test data target names:",data_test.target_names)

vectorizer = TfidfVectorizer()
data_train_vectors = vectorizer.fit_transform(data_train.data)
data_test_vectors = vectorizer.transform(data_test.data)

#Train data type
print (type(data_train_vectors.data))
print (type(data_train.target))
print (data_test_vectors.shape)

# Test data type
print (type(data_train_vectors.data))
print (type(data_train.target))
data_train_vectors.shape

#store training feature matrix in "Xtr"
Xtr = data_train_vectors
print ("Xtr:\n", Xtr)

# store training response vector in "ytr"
ytr = data_train.target
print ("ytr:",ytr)
print(ytr.shape)
np.unique(ytr)

from sklearn.naive_bayes import MultinomialNB

Xtt = data_test_vectors
print ("Xtt:\n", Xtt)

# store testing response vector in "ytt"
ytt = data_test.target
print ("ytt:",ytt)
ytt.shape
np.unique(ytt)

# Implementing classification model- using MultinomialNB

# Instantiate the estimator
clf_MNB = MultinomialNB(alpha=.01)

# Fit the model with data (aka "model training")
clf_MNB.fit(Xtr, ytr)

# Predict the response for a new observation
y_pred = clf_MNB.predict(Xtt)
print ("Predicted Class Labels:",y_pred)

# Predict the response score for a new observation
y_pred_score_mnb = clf_MNB.predict_proba(Xtt)
print("Predicted Score:\n",y_pred_score_mnb)

from sklearn.metrics import f1_score

f1_score(ytt, y_pred, average='macro')

from sklearn.neighbors import KNeighborsClassifier

# Instantiate the estimator
clf_knn =  KNeighborsClassifier(n_neighbors=5)

# Fit the model with data (aka "model training")
clf_knn.fit(Xtr, ytr)

# Predict the response for a new observation
y_pred = clf_knn.predict(Xtt)
print ("Predicted Class Labels:",y_pred)

# Predict the response score for a new observation
y_pred_score_knn = clf_knn.predict_proba(Xtt)
print ("Predicted Score:\n",y_pred_score_knn)

f1_score(ytt, y_pred, average='macro')

# Implementing Rocchio Classification
from sklearn.neighbors import NearestCentroid

# Instantiate the estimator
clf_rocchio =  NearestCentroid()

# Fit the model with data (aka "model training")
clf_rocchio.fit(Xtr, ytr)

# Predict the response for a new observation
y_pred = clf_rocchio.predict(Xtt)
print ("Predicted Class Labels:",y_pred)

# Predict the response score for a new observation

f1_score(ytt, y_pred, average='macro')
