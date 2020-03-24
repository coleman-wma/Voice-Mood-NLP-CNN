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

# Apply each function and put the results in a list
columns = []
for func in transform_functions:
    columns.append(eb_training["text"].apply(func))
    
# convert the meta features into an numpy array
meta = np.asarray(columns).T


# Calculate meta features for test set
# Apply each function and put the results in a list
test_columns = []
for func in transform_functions:
    test_columns.append(eb_test['text'].apply(func))
    
# convert the meta features into an numpy array
test_meta = np.asarray(test_columns).T


'''
Feature Selection
'''

def features_for_emotion_dimensions(training_labels,
                                    training_text,
                                    test_text,
                                    meta_feats,
                                    meta_feats_test):
    
    '''
    Takes training labels and text, performs feature selection, spits out the
    1000 most useful features for the labels. Apply once each for valence,
    activation and domination features. Does the same thing for the test set
    but feature selection only on the training set.
    '''

    # Create a version of Valence to binary so we can use chi-squared
    make_binary = training_labels.copy(deep=True)
    get_mean = make_binary.mean()
    make_binary[make_binary < get_mean] = 0
    make_binary[(get_mean > 0) & (make_binary > get_mean)] = 1
    
    # applying matrix to all training instances
    training_matrix = vectorizer.fit_transform(training_text)
    test_matrix = vectorizer.transform(test_text)
    print(training_matrix.shape)
    print(test_matrix.shape)
    
    # Find the 1000 most informative columns
    selector = SelectKBest(chi2, k=1000)
    selector.fit(training_matrix, make_binary)
    top_words = selector.get_support().nonzero()
    
    # Pick only the most informative columns in the data
    chi_matrix = training_matrix[:, top_words[0]]
    chi_test_matrix = test_matrix[:, top_words[0]]
    
    # collect all the features in one object
    features = np.hstack([meta_feats, chi_matrix.todense()])
    test_features = np.hstack([meta_feats_test, chi_test_matrix.todense()])
    
    return features, top_words[0], test_features


feat_train_val, feat_idx_val, feat_test_val =\
features_for_emotion_dimensions(eb_training['V'],
                                eb_training['text'],
                                eb_test['text'],
                                meta,
                                test_meta)

feat_train_act, feat_idx_act, feat_test_act =\
features_for_emotion_dimensions(eb_training['A'],
                                eb_training['text'],
                                eb_test['text'],
                                meta,
                                test_meta)

feat_train_dom, feat_idx_dom, feat_test_dom =\
features_for_emotion_dimensions(eb_training['D'],
                                eb_training['text'],
                                eb_test['text'],
                                meta,
                                test_meta)

'''
Collect all the test sets - feature selection performed on the training set only
'''

def get_test_feature_vector(test_inst, feat_idx, _test_meta):
    
    '''
    Generate a feature representation for test instances based on the feature
    selection performed on the training set.
    '''
    
    # Pick only the most informative columns in the data
    chi_matrix = test_inst['text'][:, feat_idx]
    
    # collect all the features in one object
    features = np.hstack([_test_meta, chi_matrix.todense()])
    
    return features

test_feats_val = get_test_feature_vector(eb_test, feat_idx_val, test_meta)

'''
Make Predictions
'''

from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
# import random

# setup train and dev splits
feat_train_V = feat_train_val[:8062, :]
feat_dev_V = feat_train_val[8062:, :]
feat_train_A = feat_train_act[:8062, :]
feat_dev_A = feat_train_act[8062:, :]
feat_train_D = feat_train_dom[:8062, :]
feat_dev_D = feat_train_dom[8062:, :]

# Valence Labels
lab_train_V = eb_train['V']
lab_dev_V = eb_dev['V']
lab_training_V = eb_training['V']
lab_test_V = eb_test['V']

# Activation Labels
lab_train_A = eb_train['A']
lab_dev_A = eb_dev['A']
lab_training_A = eb_training['A']
lab_test_A = eb_test['A']

# Dominance Labels
lab_train_D = eb_train['D']
lab_dev_D = eb_dev['D']
lab_training_D = eb_training['D']
lab_test_D = eb_test['D']

feat_train_V = np.nan_to_num(feat_train_V)
feat_train_A = np.nan_to_num(feat_train_A)
feat_train_D = np.nan_to_num(feat_train_D)

def print_preds_errors_plots(preds, labels, plot_title):
    '''
    Calculate errors and correlations. Run a plot.
    '''
    # comparing predicted with actual
    print('Prediction error: ', sum(abs(preds - labels)) / len(preds))
    
    # average of all Valence ratings
    average_all_measure = sum(labels)/len(labels)
    print('Mean error: ', sum(abs(average_all_measure - labels)) / len(preds))
    
    # calculate pearson's correlation
    corr, _ = pearsonr(preds, labels)
    print('Pearsons correlation: %.3f' % corr)
    
    # calculate spearman's correlation
    corr, _ = spearmanr(preds, labels)
    print('Spearmans correlation: %.3f' % corr)
    
    #sns.distplot(preds)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12,12))
    fig = sns.regplot(x=preds, y=labels, label=plot_title)
    #fig.set_axis_labels(xlabel='Predictions', ylabel='Actual')
    
    plt.title(plot_title)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.xlim((0.5, 4.5))
    plt.ylim((0.5, 4.5))
    plt.show(fig)


'''
VALENCE
'''
# Run the regression and generate predictions for the dev set
reg_V = Ridge(alpha=.1)
reg_V.fit(feat_train_V, lab_train_V)
predictions_V = reg_V.predict(feat_dev_V)
# Print prediction error and plot distribution - dev
print_preds_errors_plots(predictions_V, lab_dev_V, "VALENCE")

# Test set
reg_V = Ridge(alpha=.1)
reg_V.fit(feat_train_val, lab_training_V)
predictions_V = reg_V.predict(feat_test_val)
# Print prediction error and plot distribution - test
print_preds_errors_plots(predictions_V, lab_test_V, "VALENCE")


'''
ACTIVATION
'''
# Run the regression and generate predictions for the dev set
reg_A = Ridge(alpha=.1)
reg_A.fit(feat_train_A, lab_train_A)
predictions_A = reg_A.predict(feat_dev_A)
# Print prediction error and plot distribution
print_preds_errors_plots(predictions_A, lab_dev_A, "ACTIVATION")

# Test set
reg_A = Ridge(alpha=.1)
reg_A.fit(feat_train_act, lab_training_A)
predictions_A = reg_A.predict(feat_test_act)
# Print prediction error and plot distribution - test
print_preds_errors_plots(predictions_A, lab_test_A, "ACTIVATION")


'''
DOMINANCE
'''
# Run the regression and generate predictions for the dev set
reg_D = Ridge(alpha=.1)
reg_D.fit(feat_train_D, lab_train_D)
predictions_D = reg_D.predict(feat_dev_D)
# Print prediction error and plot distribution
print_preds_errors_plots(predictions_D, lab_dev_D, "DOMINANCE")

# Test set
reg_D = Ridge(alpha=.1)
reg_D.fit(feat_train_dom, lab_training_D)
predictions_D = reg_D.predict(feat_test_dom)
# Print prediction error and plot distribution
print_preds_errors_plots(predictions_D, lab_test_D, "DOMINANCE")


'''
SAVE MODELS
'''

from joblib import dump, load
dump(reg_V, 'model_text_valence.joblib')
dump(reg_A, 'model_text_activation.joblib')
dump(reg_D, 'model_text_dominance.joblib')

'''
LOAD MODELS
'''

# =============================================================================
# reg_V = load('model_text_valence.joblib')
# reg_A = load('model_text_activation.joblib')
# reg_D = load('model_text_dominance.joblib')
# =============================================================================

'''
Testing a random sentence
'''

testing_input = "I felt terrible and I didn't know what to do"
testing_input = "He should be ashamed of himself. I wanted to strangle him."
testing_input = pd.Series(testing_input)

# Calculate meta features for data input
# Apply each function and put the results in a list
test_input_columns = []
for func in transform_functions:
    test_input_columns.append(testing_input.apply(func))
    
# convert the meta features into an numpy array
test_input_meta = np.asarray(test_input_columns).T

# Convert the input test to a vector
string_matrix = vectorizer.transform(testing_input)

# Use the same set of features as learned in our model
# 'top_words' has been declared in a global scope previously
# But it may be scoped for valence and not the other two dimensions - check
chi_test_string = string_matrix[:, top_words[0]]

# collect all the features in one object
new_input_features = np.hstack([test_input_meta, chi_test_string.todense()])

print("VALENCE SCORE PREDICTION: ", reg_V.predict(new_input_features))
print("ACTIVATION SCORE PREDICTION: ", reg_A.predict(new_input_features))
print("DOMINANCE SCORE PREDICTION: ", reg_D.predict(new_input_features))

'''
NEXT STEPS:
    Implement pipeline:
    https://scikit-learn.org/dev/tutorial/text_analytics/working_with_text_data.html
    Construct 'top_words' for activation and dominance dimensions
    Implement online:
    https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
    Try wordtovec (Domenico)
    Try linear regression
    Try Naive Bayes
    Try SVM
    Try Random Forest Regressor
    Try different dimension reduction methods
    Find more data
    Deep Learning models
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

# =============================================================================
# # Create a version of Valence to binary so we can use chi-squared
# val_binary = eb_training['V'].copy(deep=True)
# val_mean = val_binary.mean()
# val_binary[val_binary < val_mean] = 0
# val_binary[(val_mean > 0) & (val_binary > val_mean)] = 1
# 
# # applying matrix to all training instances
# training_matrix = vectorizer.fit_transform(eb_training['text'])
# print(training_matrix.shape)
# 
# # Find the 1000 most informative columns
# selector = SelectKBest(chi2, k=1000)
# selector.fit(training_matrix, val_binary)
# top_words = selector.get_support().nonzero()
# 
# # Pick only the most informative columns in the data
# chi_matrix = training_matrix[:, top_words[0]]
# 
# 
# 
# # collect all the features in one object
# features = np.hstack([meta, chi_matrix.todense()])
# =============================================================================


