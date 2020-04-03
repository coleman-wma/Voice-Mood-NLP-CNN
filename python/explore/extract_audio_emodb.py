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
# read in LPMS data - clamp between 0 - 20 kHz - 40 bins, use first 20 only to
    cover up to 8 kHz at least
# try 50 ms windows with 25% overlap
# drop silence, select 0.4 sec chunks, define LPMS array of size 13 x ??
# (depends on temporal window)
# scale LPMS (already scaled to dB so - might not be necessary?)
"""

#%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display

import os
import pandas as pd
import numpy as np

# Storage
import pickle

# Audio functions
import librosa as lib

# Normalise data
from sklearn.preprocessing import MinMaxScaler

# Play a file
import simpleaudio as sa

'''
LOAD A FILE
'''

def load_file(file_idx, fnames, filepath):
    '''
    Load a file, return the samples
    '''
    name = fnames[file_idx]
    # file to load
    getfile = filepath + str(name)
    # get datapoints and sample rate of file and load it
    samples, sr = lib.load(getfile, sr=None)
    
    return samples, getfile


'''
PLOTTING/PLAYING FUNCTIONS
'''

def plot_raw_wave(file):
    '''
    Plot the raw wave
    '''
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(file, sr=16000)


def plot_lpms_chunk(win):
    '''
    Plot lpms instance chunk
    '''
    librosa.display.specshow(win,
                         sr=16000,
                         x_axis='s',
                         y_axis='linear',
                         hop_length=256)
    plt.colorbar(format='%+2.0f dB')


def play_file(file):
    '''
    Play a file
    '''
    wave_obj = sa.WaveObject.from_wave_file(file)
    play_obj = wave_obj.play()
    # Wait until sound has finished playing
    play_obj.wait_done()


'''
DELETE THE SILENCE
'''

def strip_silence(file_to_strip):
    '''
    Takes a non-treated file and strips silent segments from it - provides an
    array as final output
    '''
    intervals = librosa.effects.split(file_to_strip,
                                      top_db=30,
                                      frame_length=1024,
                                      hop_length=256)
    
    # compile non silent parts to a list of lists
    non_silent = []
    for i in range(intervals.shape[0]):
        chunk = file_to_strip[intervals[i][0]:intervals[i][1]]
        non_silent.append(chunk)
    
    # flatten list of lists to a single list
    non_silent_arr = [item for sublist in non_silent for item in sublist]
    
    return np.asarray(non_silent_arr)


'''
CONVERT TO LPMS
'''

def convert_to_lpms(raw_silenced):
    '''
    Take a raw wave with the silence removed and convert to a LPMS matrix
    '''
    log_pow = librosa.feature.melspectrogram(y=raw_silenced,
                                             sr=16000,
                                             n_mels=40, 
                                             win_length=1024,
                                             hop_length=256)
    
    # scale to dB
    log_pow_db = librosa.power_to_db(log_pow, ref=np.max)
    
    return log_pow_db


'''
CHUNK THE FILE TO 0.4 sec CHUNKS

SCALE THE CONTENTS

SAVE INDIVIDUAL INSTANCES WITH APPROPRIATE LABELS
'''

def chunk_scale_lpms_matrix(lpms_matrix, fname):
    '''
    Split the LPMS matrix up into 0.4 second chunks
    '''
    
    # to track chunks
    track_start = 0
    track_end = 25
    
    # lists to hold data and labels
    data_scaled_list = []
    labels_list = []
    
    # define scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # taking the lower 20 bins
    # step through the instance extracting 0.4 sec length chunks
    while track_end < lpms_matrix.shape[1]:
        
        # get window data
        win = lpms_matrix[20:40, track_start:track_end]
        
        # scale the data
        win_scaled = scaler.fit_transform(win)
        
        # append data and labels
        data_scaled_list.append(win_scaled)
        labels_list.append(fname[5])
        
        # increment start and end of chunks
        track_start += 1
        track_end += 1
    
    return data_scaled_list, labels_list


###################################
########## WORKINGS ###############
###################################

# get emodb filenames
filenames = os.listdir("/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/emodb/wav")

# filename and path
path = '/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/emodb/wav/'

# create empty series to hold data
lpms_scaled_chunks = []
lpms_scaled_labels = []


for f in range(len(filenames)):
    
    samples, getfile = load_file(f, filenames, path)
    
    silence_stripped = strip_silence(samples)
    
    lpms_ified = convert_to_lpms(silence_stripped)
    
    data_scaled, labels = chunk_scale_lpms_matrix(lpms_ified, filenames[f])
    
    for d in data_scaled:
        lpms_scaled_chunks.append(d)
    
    for l in labels:
        lpms_scaled_labels.append(l)
    
    print(f)

    
    
########################################
########## TABLE MUNGING ###############
########################################

# to pick out english emotion names
emotions_eng = ['fear', 'disgust', 'happy', 'bored', 'neutral', 'sad', 'angry']
emotions_code = ['A', 'E', 'F', 'L', 'N', 'T', 'W']

# to assign VAD values - taken from Russell & Mahrabian (1977)
vals_emos_V = [-0.64, -0.6, 0.81, -0.65, 0, -0.63, -0.51]
vals_emos_A = [0.6, 0.35, 0.51, -0.62, 0, -0.27, 0.59]
vals_emos_D = [-0.43, 0.11, 0.46, -0.33, 0, -0.33, 0.25]

# to hold values
vec_eng = []
vec_V = []
vec_A = []
vec_D = []

# to search for german emotion tag
unique_ger = np.unique(lpms_scaled_labels).tolist()

# step through all instances, append values to lists
for s in range(0, len(lpms_scaled_labels)):
    find = lpms_scaled_labels[s]
    idx = unique_ger.index(find)
    vec_eng.append(emotions_eng[idx])
    vec_V.append(vals_emos_V[idx])
    vec_A.append(vals_emos_A[idx])
    vec_D.append(vals_emos_D[idx])


audiofile_metadata = pd.DataFrame([lpms_scaled_labels,
                                   vec_eng,
                                   vec_V,
                                   vec_A,
                                   vec_D]).T

audiofile_metadata.columns = ["ger", "eng", "V", "A", "D"]


########################################
########## PICKLE DATA #################
########################################

with open('/Volumes/COLESLAW_1TB/BELL_LABS/emodb_labels.data',
          'wb') as new_data:
    # store the data as binary data stream - protocol 2 is so it can be used on Kevin Street
    pickle.dump(audiofile_metadata, new_data, protocol=4)
    print('Saved labels')

with open('/Volumes/COLESLAW_1TB/BELL_LABS/emodb_audio.data',
          'wb') as new_data:
    # store the data as binary data stream - protocol 2 is so it can be used on Kevin Street
    pickle.dump(lpms_scaled_chunks, new_data, protocol=4)
    print('Saved data')
