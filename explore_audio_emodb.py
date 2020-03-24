#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 17:19:23 2020

@author: billcoleman

Exploring and data munging for the emodb database
.wav audio files labelled for emotions

Based on:
    https://towardsdatascience.com/building-a-vocal-emotion-sensor-with-deep-learning-bedd3de8a4a9
# read file data - name as string, split string for emotion category
# set VAD values per emotion
# read in LPMS data - clamp between 0 - 20 kHz - 26 bins, use first 13 only
# try 50 ms windows with 25% overlap
# drop silence, select 0.4 sec chunks, define LPMS array of size 13 x ??
# (depends on temporal window)
# scale LPMS (already scaled to dB so - might not be necessary?)
"""

import os
import pandas as pd
import numpy as np

# get emodb filenames
filenames = os.listdir("/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/emodb/wav")

# the 6th element in each string denotes the emotion category
emo_cats = []
for f in filenames:
    emo_cats.append(f[5])

# make a dataframe that will hold filename, emotional category and VAD values
audiofile_metadata = pd.DataFrame([filenames, emo_cats]).T
audiofile_metadata.columns = ["filename", "ger_cat"]

# to pick out english emotion names
eng_emotions = ['fear', 'disgust', 'happy', 'bored', 'neutral', 'sad', 'angry']

# to assign VAD values - taken from Russell & Mahrabian (1977)
vals_emos_V = [-0.64, -0.6, 0.81, -0.65, 0, -0.63, -0.51]
vals_emos_A = [0.6, 0.35, 0.51, -0.62, 0, -0.27, 0.59]
vals_emos_D = [-0.43, 0.11, 0.46, -0.33, 0, -0.33, 0.25]

# to hold values
eng_cat = []
vec_V = []
vec_A = []
vec_D = []

# to search for german emotion tag
unique_ger = np.unique(audiofile_metadata['ger_cat']).tolist()

# step through all instances, append values to lists
for s in range(0, audiofile_metadata.shape[0]):
    # for e in np.unique(audiofile_metadata['ger_cat']):
    find = audiofile_metadata['ger_cat'][s]
    idx = unique_ger.index(find)
    eng_cat.append(eng_emotions[idx])
    vec_V.append(vals_emos_V[idx])
    vec_A.append(vals_emos_A[idx])
    vec_D.append(vals_emos_D[idx])
    
# bring it all together in the dataframe
audiofile_metadata['eng_cat'] = eng_cat
audiofile_metadata['V'] = vec_V
audiofile_metadata['A'] = vec_A
audiofile_metadata['D'] = vec_D