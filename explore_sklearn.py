#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:23:12 2020

@author: billcoleman
"""

import pandas as pd
eb = pd.read_csv("data/emobank.csv", index_col=0)

print(eb.shape)
test = eb.head()

# Visualisations

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

val_objects = np.unique(eb['V'])
val_y_pos = np.arange(len(val_objects))
val_counts = np.unique(eb['V'], return_counts=True)

plt.figure(figsize=(12,6))
plt.bar(val_y_pos, val_counts[1], align='center', alpha=0.5)
plt.xticks(val_y_pos, val_counts[0], rotation=90)
plt.ylabel('Count')
plt.title('Valence Distribution')
plt.show()


act_objects = np.unique(eb['A'])
act_y_pos = np.arange(len(act_objects))
act_counts = np.unique(eb['A'], return_counts=True)

plt.figure(figsize=(12,6))
plt.bar(act_y_pos, act_counts[1], align='center', alpha=0.5)
plt.xticks(act_y_pos, act_counts[0], rotation=90)
plt.ylabel('Count')
plt.title('Activation Distribution')
plt.show()


dom_objects = np.unique(eb['D'])
dom_y_pos = np.arange(len(dom_objects))
dom_counts = np.unique(eb['D'], return_counts=True)

plt.figure(figsize=(12,6))
plt.bar(dom_y_pos, dom_counts[1], align='center', alpha=0.5)
plt.xticks(dom_y_pos, dom_counts[0], rotation=90)
plt.ylabel('Count')
plt.title('Dominance Distribution')
plt.show()

sns.distplot(val_objects)
sns.distplot(act_objects)
sns.distplot(dom_objects)

with sns.axes_style("white"):
    sns.jointplot(x=eb['V'], y=eb['A'], kind="hex", color="k")
    
with sns.axes_style("white"):
    sns.jointplot(x=eb['V'], y=eb['D'], kind="hex", color="k")

with sns.axes_style("white"):
    sns.jointplot(x=eb['D'], y=eb['A'], kind="hex", color="k")

sns.jointplot(x=eb['V'], y=eb['A'], data=eb, kind="kde")
sns.jointplot(x=eb['V'], y=eb['D'], data=eb, kind="kde")
sns.jointplot(x=eb['D'], y=eb['A'], data=eb, kind="kde")

ratings = pd.DataFrame([eb['V'], eb['A'], eb['D']]).T

sns.pairplot(ratings)

'''
Below mainly taken from:
    https://www.dataquest.io/blog/natural-language-processing-with-python/
'''

from sklearn.feature_extraction.text import CountVectorizer

# outline code on a small data subset
headlines = eb['text'][:10]

# Construct a bag of words matrix
# This will lowercase everything and ignore all punctuation by default
# It will also remove stop words
vectorizer = CountVectorizer(lowercase=True, stop_words="english")

matrix = vectorizer.fit_transform(headlines)
# A bag of words vectorizer with just a few lines of code
print(matrix.todense())

# applying this to all instances
full_matrix = vectorizer.fit_transform(eb['text'])
print(full_matrix.shape)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import re

# split into train, validate and test splits
eb_train = eb.loc[(eb.split == 'train')]
eb_dev = eb.loc[(eb.split == 'dev')]
eb_test = eb.loc[(eb.split == 'test')]
eb_training = pd.concat([eb_train, eb_dev])


# Create a version of Valence to binary so we can use chi-squared
val_binary = eb_training['V'].copy(deep=True)
val_mean = val_binary.mean()
val_binary[val_binary < val_mean] = 0
val_binary[(val_mean > 0) & (val_binary > val_mean)] = 1

# applying matrix to all training instances
training_matrix = vectorizer.fit_transform(eb_training['text'])
print(training_matrix.shape)

# Find the 1000 most informative columns
selector = SelectKBest(chi2, k=1000)
selector.fit(training_matrix, val_binary)
top_words = selector.get_support().nonzero()

# Pick only the most informative columns in the data
chi_matrix = training_matrix[:, top_words[0]]

'''
Adding meta features
'''

transform_functions = [
        lambda x: len(x),                                   # 0
        lambda x: x.count(" "),                             # 1
        lambda x: x.count("."),                             # 2
        lambda x: x.count("!"),                             # 3
        lambda x: x.count("?"),                             # 4
        lambda x: len(x) / (x.count(" ") + 1),              # 5
        lambda x: x.count(" ") / (x.count(".") + 1),        # 6
        lambda x: len(re.findall("d", x)),                  # 7
        lambda x: len(re.findall("[A-Z]", x))               # 8
        ]

# Apply each fiunction and put the results in a list
columns = []
for func in transform_functions:
    columns.append(eb_training["text"].apply(func))
    
# convert the meta features into an numpy array
meta = np.asarray(columns).T

# collect all the features in one object
features = np.hstack([meta, chi_matrix.todense()])

'''
Make Predictions
'''

from sklearn.linear_model import Ridge
# import random

# setup train and dev splits
feat_train = features[:8062, :]
feat_dev = features[8062:, :]

# Valence Labels
lab_train_V = eb_train['V']
lab_dev_V = eb_dev['V']
lab_test_V = eb_test['V']

# Activation Labels
lab_train_A = eb_train['A']
lab_dev_A = eb_dev['A']
lab_test_A = eb_test['A']

# Dominance Labels
lab_train_D = eb_train['D']
lab_dev_D = eb_dev['D']
lab_test_D = eb_test['D']

feat_train = np.nan_to_num(feat_train)

# Run the regression and generate predictions for the dev set
reg_V = Ridge(alpha=.1)
reg_V.fit(feat_train, lab_train_V)
predictions_V = reg_V.predict(feat_dev)

reg_A = Ridge(alpha=.1)
reg_A.fit(feat_train, lab_train_A)
predictions_A = reg_A.predict(feat_dev)

reg_D = Ridge(alpha=.1)
reg_D.fit(feat_train, lab_train_D)
predictions_D = reg_D.predict(feat_dev)

def print_preds_errors_plots(preds, labels):
    # comparing predicted with actual
    print(sum(abs(preds - labels)) / len(preds))
    
    # average of all Valence ratings
    average_all_measure = sum(labels)/len(labels)
    print(sum(abs(average_all_measure - labels)) / len(preds))
    
    sns.distplot(preds)

# Print prediction error and plot distribution
print_preds_errors_plots(predictions_V, lab_dev_V)
print_preds_errors_plots(predictions_A, lab_dev_A)
print_preds_errors_plots(predictions_D, lab_dev_D)

'''
Note: Features selected above as being most informative for valence
Repeat that exercise for Activation and Dominance and likely those scores will
improve.
'''

# =============================================================================
# from collections import Counter
# 
# headlines = eb['text'][:10]
# 
# # find all the unique words in the subset
# unique_words = list(set(" ".join(headlines).split(" ")))
# 
# def make_matrix(hlines, vocab):
#     matrix = []
#     for headline in hlines:
#         # Count each word in the headline and make a dictionary
#         counter = Counter(headline)
#         # Turn the dictionary into a matrix row using the vocab
#         row = [counter.get(w, 0) for w in vocab]
#         matrix.append(row)
#     df = pd.DataFrame(matrix)
#     df.columns = unique_words
#     return df
# 
# print(make_matrix(headlines, unique_words))
# 
# 
# import re
# 
# # Lowercase, then replace any non-letter, space or digit character in the headlines
# #new_headlines = [re.sub(r'[^wsd]','',h.lower()) for h in headlines]
# 
# new_headlines = [re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
#                         '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', h.lower()) for h in headlines]
# new_headlines = [re.sub("(@[A-Za-z0-9_]+)","", h.lower()) for h in headlines]
# new_headlines = [re.sub(r'[,@\'?\.$%_:\-"’“”]', "", h.lower(), flags=re.I) for h in headlines]
# 
# # Replace sequences of whitespace with a space character
# new_headlines = [re.sub("\s+", " ", h) for h in new_headlines]
# 
# unique_words = list(set(" ".join(new_headlines).split(" ")))
# # We've reduced the number of columns in the matrix a little
# print(make_matrix(new_headlines, unique_words))
# =============================================================================
