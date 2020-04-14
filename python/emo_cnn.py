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


'''
OPEN FILES
'''

# LOAD
with open('/Volumes/COLESLAW_1TB/BELL_LABS/emodb_audio.data', 'rb') as new_data:
    all_data = pickle.load(new_data)

# LOAD
with open('/Volumes/COLESLAW_1TB/BELL_LABS/emodb_labels.data', 'rb') as new_data:
    all_labels = pickle.load(new_data)

'''
TRAIN AND TEST SETS
'''

X_train, X_test, y_train, y_test = train_test_split(all_data,
                                                    all_labels,
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


EPOCHS = 5

mod_history = model.fit(
        X_train, y_train, epochs=EPOCHS, validation_split = 0.2,
        verbose=1, callbacks=[tfdocs.modeling.EpochDots()]
        )

mod_evaluate = model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict(X_test, batch_size=32, verbose=2)

