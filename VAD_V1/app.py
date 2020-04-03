#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:26:54 2020

@author: billcoleman

From: https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
"""

from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import re
from sklearn.linear_model import Ridge

# =============================================================================
# import pickle
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
# =============================================================================



app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

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

    # Declare vectorizer
    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
        
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

@app.route('/result', methods=['POST'])
def predict(prediction=None):
    
    eb = pd.read_csv("data/emobank.csv", index_col=0)
    
    # split into train, validate and test splits
    eb_train = eb.loc[(eb.split == 'train')]
    eb_dev = eb.loc[(eb.split == 'dev')]
    eb_test = eb.loc[(eb.split == 'test')]
    eb_training = pd.concat([eb_train, eb_dev, eb_test])
    
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
    
    #####################
    #####################
    #####################

    if request.method == 'POST':
        message = request.form['message']
        data = pd.DataFrame([message])  # [message]
        # Calculate meta features for test set
        # Apply each function and put the results in a list
        test_columns = []
        for func in transform_functions:
            test_columns.append(data[0].apply(func))
        
        # convert the meta features into an numpy array
        test_meta = np.asarray(test_columns).T
        
        feat_train_val, feat_idx_val, feat_test_val =\
        features_for_emotion_dimensions(eb_training['V'],
                                eb_training['text'],
                                data[0],
                                meta,
                                test_meta)
        
        lab_training_V = eb_training['V']
        feat_train_V = np.nan_to_num(feat_train_val)
        
        reg_V = Ridge(alpha=.1)
        reg_V.fit(feat_train_V, lab_training_V)
        predictions_V = reg_V.predict(feat_test_val)
        
        return render_template('result.html', prediction=predictions_V)

if __name__ == '__main__':
	app.run(debug=True)