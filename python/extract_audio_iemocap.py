#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:46:51 2020

@author: billcoleman

Extract all the .wav files from the IEMOCAP database and prepare them for
training with the CNN

Isolate each file
Convert to LPMS (better representation for audio deep learning)
Remove silence
Chop into 0.4 second chunks
Final representation will be nRows,nCols,nDims = 20, 25, 1
Total number of instances worked out to be 2,146,661
"""

import os

# Storage
import pickle

# Audio functions
import librosa as lib

# Normalise data
from sklearn.preprocessing import MinMaxScaler

# include extract_audio_emodb for munging functions
import extract_audio_emodb

# paths to text files - location of VAD labels
# Change session numbers for different folders, Sessions 1 - 5

'''
MUNGING FUNCTIONS
'''
# strip_silence - returns np.asarray(non_silent_arr)
# extract_audio_emodb.strip_silence(file_to_strip)

# comvert to lpms - returns LPMS 40 mels
# extract_audio_emodb.convert_to_lpms(raw_silenced)

'''
LOAD A FILE
'''

def load_file(filepath):
    '''
    Load a file, return the samples
    '''
    
    # get datapoints and sample rate of file and load it
    samples, sr = lib.load(filepath, sr=None)
    
    return samples


'''
CHUNK THE FILE TO 0.4 sec CHUNKS

SCALE THE CONTENTS

SAVE INDIVIDUAL INSTANCES WITH APPROPRIATE LABELS
'''

def chunk_scale_lpms_matrix(lpms_matrix, data_list, names_list, name):
    '''
    Split the LPMS matrix up into 0.4 second chunks
    '''
    
    # to track chunks
    track_start = 0
    track_end = 25
    
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
        data_list.append(win_scaled)
        names_list.append(name)
        
        # increment start and end of chunks
        track_start += 1
        track_end += 1


'''
Read in the wav filenames so we can reference labels
'''

sess = [1, 2, 3, 4, 5]

wav_folder = "/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/IEMOCAP_full_release/Session1/sentences/wav"

wav_names = []
wav_samples = []

for digit in sess:
    wav_folder = "/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/IEMOCAP_full_release/Session" + str(digit) + "/sentences/wav"
    print(wav_folder)

    # for every folder in wav_folder
    for f in os.listdir(wav_folder):
        if f == ".DS_Store":
            continue
        # for every file in each folder
        # print(f)
        for e in os.listdir(str(wav_folder + '/' + f)):
            # if the file extension is .wav
            if e[-4:] != ".wav":
                continue
    
            # Make path to file
            file_to_load = str(wav_folder + '/' + f + '/' + e)
            # Read the file samples
            file_samples = load_file(file_to_load)
            # Strip silence
            silence_stripped = extract_audio_emodb.strip_silence(file_samples)
            # Return LPMS 40 mels, 16 kHz
            lpms_ified = extract_audio_emodb.convert_to_lpms(silence_stripped)
            # Scaled the data, chunk to 0.4 secs
            chunk_scale_lpms_matrix(lpms_ified, wav_samples, wav_names, e)
            
        print("Length of wav_names: ", len(wav_names))
        print("Length of wav_samples: ", len(wav_samples))


with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_labels.data',
          'wb') as new_data:
    # store the data as binary data stream - protocol 2 is so it can be used on Kevin Street
    pickle.dump(wav_names, new_data, protocol=4)
    print('Saved labels')

with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_audio.data',
          'wb') as new_data:
    # store the data as binary data stream - protocol 2 is so it can be used on Kevin Street
    pickle.dump(wav_samples, new_data, protocol=4)
    print('Saved data')


with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_labels.data',
          'rb') as filehandle:
    # read the data as binary data stream
    wav_names = pickle.load(filehandle)

with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_audio.data',
          'rb') as filehandle:  
    # read the data as binary data stream
    wav_samples = pickle.load(filehandle)
