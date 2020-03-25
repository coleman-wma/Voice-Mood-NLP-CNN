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
emotions_code = ['A', 'E', 'F', 'L', 'N', 'T', 'W']

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

'''
EXTRACTING AUDIO FILES
'''

# Storage
import pickle

# Audio functions
import librosa as lib
import librosa.display

# Normalise data
from sklearn.preprocessing import MinMaxScaler

# filename and path
path = '/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/emodb/wav/'

# create empty series to hold data
lpms = []
stim_id = []

test = filenames[300]
# file to load
getfile = path + str(test)
# get datapoints and sample rate of file and load it
y, sr = lib.load(getfile, sr=None)

'''
Plot wave
'''
plt.figure(figsize=(14, 5))
librosa.display.waveplot(y, sr=sr)

'''
Play a file
'''
import simpleaudio as sa
#filename = 'myfile.wav'
wave_obj = sa.WaveObject.from_wave_file(getfile) # getfile
play_obj = wave_obj.play()
play_obj.wait_done()  # Wait until sound has finished playing

'''
Index to non-silent segments
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
        chunk = y[intervals[i][0]:intervals[i][1]]
        non_silent.append(chunk)
    
    # flatten list of lists to a single list
    non_silent_arr = [item for sublist in non_silent for item in sublist]
    
    return np.asarray(non_silent_arr)

de_silenced = strip_silence(y)

# plot the figure to check
plt.figure(figsize=(14, 5))
librosa.display.waveplot(de_silenced, sr=sr)

'''
Take each de-silenced file
Convert to lpms
Chop into 0.4 second chunks
Store these as instances with appropriate label
'''

log_pow = librosa.feature.melspectrogram(y=de_silenced,
                                         sr=16000,
                                         n_mels=40, 
                                         win_length=1024,
                                         hop_length=256)

# scale to dB
log_pow_db = librosa.power_to_db(log_pow, ref=np.max)

# to pick out a specific window
win = log_pow_db[20:40,0:25]

plt.figure(figsize=(14, 5))
librosa.display.specshow(win,  # log_pow_db,  # win
                         sr=16000,
                         x_axis='s',
                         y_axis='linear',
                         hop_length=256)
plt.colorbar(format='%+2.0f dB')

# divide into instances 0.4 seconds long (25 x hop_length)
track_end = 25

all_data = []
all_labels = []

# taking the lower 20 bins
# step through the instance extracting 0.4 sec length chunks
while track_end < log_pow_db.shape[1]:
    win = log_pow_db[20:40, 0:track_end]
    all_data.append(win)
    all_labels.append('N')
    print(track_end)
    track_end += 1




# for each file in the test folder
for f in filenames:
    # file to load
    getfile = path + str(f)
    # get datapoints and sample rate of file and load it
    y, sr = lib.load(getfile, sr=None)

    # EXTRACT LPMS and LPMS delta data
    # using a window length of 1024 samples at 16kHz would mean a window length
    # of 64 msecs
    log_pow = librosa.feature.melspectrogram(y=y,
                                             sr=16000,
                                             n_mels=40, 
                                             win_length=1024,
                                             hop_length=256)
    # scale to dB
    # log_pow_db = librosa.power_to_db(log_pow, ref=np.max)
    log_pow_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)),
                                         ref=np.max)

    # append data to series
    lpms.append(log_pow_db)

    # index
    stim_id.append(f)
    
    # mark progress
    ct = f - 1000000
    if ct % 100 == 0:
        print(ct / 100, "% done")
        

y, sr = librosa.load(getfile, sr=None)
plt.figure(figsize=(12, 8))

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

plt.subplot(4, 2, 2)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
plt.subplot(4, 2, 3)
librosa.display.specshow(CQT, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

plt.subplot(4, 2, 4)
librosa.display.specshow(CQT, y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (Hz)')

C = librosa.feature.chroma_cqt(y=y, sr=sr)
plt.subplot(4, 2, 5)
librosa.display.specshow(C, y_axis='chroma')
plt.colorbar()
plt.title('Chromagram')

plt.subplot(4, 2, 6)
librosa.display.specshow(D, cmap='gray_r', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear power spectrogram (grayscale)')

plt.subplot(4, 2, 7)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log power spectrogram')

plt.subplot(4, 2, 8)
Tgram = librosa.feature.tempogram(y=y, sr=sr)
librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
plt.colorbar()
plt.title('Tempogram')
plt.tight_layout()

plt.show()