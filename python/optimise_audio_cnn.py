#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:09:34 2020

@author: billcoleman

Following based on: https://www.tensorflow.org/tutorials/keras/regression
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

# Storage
import pickle
# To save models
from sklearn.externals import joblib


'''
OPEN FILES
'''

# LOAD
with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_audio.data', 'rb') as new_data:
    all_data = pickle.load(new_data)

# LOAD
with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_labels.data', 'rb') as new_data:
    all_labels = pickle.load(new_data)

# LOAD
with open('/Volumes/COLESLAW_1TB/BELL_LABS/unique_wavs_VAD.data', 'rb') as new_data:
    all_unique_labels = pickle.load(new_data)

# Pick out VAD labels for each unique .wav
all_val = []
all_act = []
all_dom = []

for e in range(len(all_labels)):
    # for every unique .wav
    lookup = all_labels[e][:-4]
    # pick out the label and append it to a list
    all_val.append(all_unique_labels.loc[lookup]['label_val'])
    all_act.append(all_unique_labels.loc[lookup]['label_act'])
    all_dom.append(all_unique_labels.loc[lookup]['label_dom'])
    
    if e % 100:
        comp = e / (all_unique_labels.shape[0] * 0.01)
        print("DONE % = ", comp)

# Make dataframe of all individual wavs and corresponding labels
all_labels_df = pd.DataFrame([all_labels, all_val, all_act, all_dom]).T
all_labels_df.columns = ['wav_name', 'V', 'A', 'D']


'''
TRAIN AND TEST SETS
'''

X_train, X_test, y_train, y_test = train_test_split(all_data,
                                                    all_labels_df,
                                                    test_size=0.2,
                                                    random_state=42)

'''
RESHAPE
'''

nRows,nCols,nDims = 20, 25, 1
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_train = X_train.reshape(X_train.shape[0], nRows, nCols, nDims)
X_test = X_test.reshape(X_test.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, nDims)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Labels for Valence only
y_train = y_train['V']
y_test = y_test['V']
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
y_train = np.expand_dims(y_train, -1)
y_test = np.expand_dims(y_test, -1)

# Labels for VAD
y_train = np.asarray([y_train['V'], y_train['A'], y_train['D']]).T
y_test = np.asarray([y_test['V'], y_test['A'], y_test['D']]).T
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')
# =============================================================================
# y_train = np.expand_dims(y_train, -1)
# y_test = np.expand_dims(y_test, -1)
# =============================================================================

'''
BUILD MODEL
'''

# Create Model
# =============================================================================
# def createModel():
#     model = tf.keras.models.Sequential()
#     
#     model.add(tf.keras.layers.Conv2D(12, (5, 5), padding='same', activation='relu',
#                      strides=(2,2), input_shape=input_shape))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.Dropout(0.2))
#     model.add(tf.keras.layers.Conv2D(24, (3, 3), padding='same', activation='relu',
#                      strides=(1,1)))
#     model.add(tf.keras.layers.BatchNormalization())
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 
#     # Pooling
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(1, activation='linear'))
#     
#     optimizer = tf.keras.optimizers.RMSprop(0.001)
# 
#     model.compile(loss='mse',
#                   optimizer=optimizer,
#                   metrics=['mae', 'mse'])
#     
#     return model
# =============================================================================

def createModel():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=input_shape),
    layers.Dense(64, activation='relu'),
    layers.Flatten(),
    layers.Dense(3)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    
    return model

model = createModel()

model.summary()

example_batch = X_train[:10]
example_result = model.predict(example_batch)
example_result


EPOCHS = 2

mod_history = model.fit(
        X_train, y_train, epochs=EPOCHS, validation_split = 0.2,
        verbose=1, callbacks=[tfdocs.modeling.EpochDots()]
        )

mod_evaluate = model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict(X_test, batch_size=32, verbose=2)

from scipy.stats import pearsonr, spearmanr
# visualisations
import matplotlib.pyplot as plt
import seaborn as sns

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
    plt.xlim((0, 5.5))
    plt.ylim((0, 5.5))
    plt.show(fig)


# Print prediction error and plot distribution - test
print_preds_errors_plots(y_pred[:, 0], y_test[:, 0], "VALENCE")
print_preds_errors_plots(y_pred[:, 1], y_test[:, 1], "ACTIVATION")
print_preds_errors_plots(y_pred[:, 2], y_test[:, 2], "DOMINANCE")


# SAVE MODEL IN ONE ENTITY
# Export the model to a SavedModel
model.save('model_audio_iemocap_v2.h5')


# SAVE MODEL ARCHITECTURE AND WEIGHTS SEPERATELY
# serialize model to JSON
model_json = model.to_json()
with open("model_audio_iemocap.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_audio_iemocap.h5")
print("Saved model to disk")

from keras.models import model_from_json
# TO LOAD MODEL
# load json and create model
json_file = open('model_audio_iemocap.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_audio_iemocap.h5")
print("Loaded model from disk")

