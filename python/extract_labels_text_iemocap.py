#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:52:20 2020

@author: billcoleman
"""

import os
import pandas as pd
import numpy as np

# Storage
import pickle



'''
Step through every rating file in each folder
Pull out the instance reference
Pull out the VAD labels
Then amalgamate the ratings (average over all oracle raters and self ratings)
'''

sess = [1, 2, 3, 4, 5]

labels = []

for digit in sess:
    ev_subject = "/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/IEMOCAP_full_release/Session" + str(digit) + "/dialog/EmoEvaluation/"

    '''
    Loop through all the folders that have evaluation information, pulling all
    ratings out and storing them in a list
    '''
    
    # for every file in this folder
    for f in os.listdir(ev_subject):
        
        # if it picks this up, skip to the next
        if f == "Attribute" or f == "Self-evaluation":
    
            print("FOLDER: ", f)
            
            # there are three folders here, for each of them
            for s in os.listdir(str(ev_subject + '/' + f)):
        
                # look at each text file that ends like this
                # if the file extension is atr.txt
                if s[-7:] != "atr.txt":
                    continue
                print("FILE: ", s)
                
                # open the file
                file = open(str(ev_subject + '/' + f + '/' + s))
                
                # Read each line in the file and append to an object
                while True:
                    
                    # while there's a line to read, read it
                    this_line = file.readline()
                    
                    # if we've reached the end of the file
                    if not this_line:
                        break
                    
                    # append the line to the list
                    labels.append(this_line)

label_split = []
label_tags = []
label_val = []
label_act = []
label_dom = []
# Split data by :
# Step through the list of labels
for i in range(len(labels)):
    # split the instance by :
    splits = labels[i].split(":")
    # append to a list
    label_split.append(splits)
    # split the tag by whitespace then append to list
    label_tags.append(splits[0].split())
    # Get val label
    label_val.append(float(splits[2][4]))
    # Get act label
    label_act.append(float(splits[1][4]))
    # Get dom label
    if (len(splits) > 3):
        label_dom.append(float(splits[3][4]))
    else:
        label_dom.append(np.NaN)


# Check the spread of labels - how many NANs in label_dom
np.unique(label_dom, return_counts=True)
np.unique(label_act, return_counts=True)
np.unique(label_val, return_counts=True)

# Replace the 3 NaNs in dom after splitting into train/test
# Make a dataframe with all this stuff
labels_df = pd.DataFrame([label_tags, label_val, label_act, label_dom, label_split, labels],
                         dtype=float).T
labels_df.columns = ['label_tags', 'label_val', 'label_act', 'label_dom', 'label_split', 'labels']

# read in the names of all individual wavs
with open('/Volumes/COLESLAW_1TB/BELL_LABS/iemocapS1_labels.data',
          'rb') as filehandle:
    # read the data as binary data stream
    wav_names = pickle.load(filehandle)

# check unique instances in wav_names and labels_tags
# How many unique instances are there?
uniq_tags_list, uniq_tags_ct = np.unique(label_tags, return_counts=True)
print("No. of unique tags: ", len(uniq_tags_list))
uniq_wavs_list, uniq_wavs_ct = np.unique(wav_names, return_counts=True)
print("No. of unique wavs: ", len(uniq_wavs_list))

# Step through labels_df and chop 'label_tags' so we can match it easily with the wavs
label_tags_str = []
for i in range(labels_df.shape[0]):
    label_tags_str.append(labels_df['label_tags'][i][0])
    
labels_df['label_tags_str'] = label_tags_str

# Step through the unique wavs list and aggregate VAD scores for each
uniq_wavs_tags = []
for i in range(len(uniq_wavs_list)):
    # Chop the '.wav' off the end of each filename
    uniq_wavs_tags.append(uniq_wavs_list[i][:-4])
    
# Make a dataframe with both of these
uniq_wavs_df = pd.DataFrame([uniq_wavs_list, uniq_wavs_tags]).T
uniq_wavs_df.columns=['uniq_wavs_list', 'uniq_wavs_tags']

# Get means of labels for instances
act_sum = labels_df.groupby('label_tags_str')['label_act'].sum()
act_count = labels_df.groupby('label_tags_str')['label_act'].count()
act_mean = act_sum / act_count
val_sum = labels_df.groupby('label_tags_str')['label_val'].sum()
val_count = labels_df.groupby('label_tags_str')['label_val'].count()
val_mean = val_sum / val_count
dom_sum = labels_df.groupby('label_tags_str')['label_dom'].sum()
dom_count = labels_df.groupby('label_tags_str')['label_dom'].count()
dom_mean = dom_sum / dom_count

# Get all labels together
labels_mean = pd.DataFrame([val_mean, act_mean, dom_mean]).T
labels_mean.columns = [val_mean, act_mean, dom_mean]

# Join tags of unique wavs to VAD averages
labels_mean_wavs = pd.DataFrame(labels_mean, index=uniq_wavs_tags)

with open('/Volumes/COLESLAW_1TB/BELL_LABS/unique_wavs_VAD.data',
          'wb') as new_data:
    # store the data as binary data stream - protocol 2 is so it can be used on Kevin Street
    pickle.dump(labels_mean_wavs, new_data, protocol=4)
    print('Saved labels')


'''
Isolating the transcriptions for each wav instance
Pull out the instance reference & transcription

Then amalgamate with ratings
'''

sess = [1, 2, 3, 4, 5]

transcriptions = []

for digit in sess:
    ev_subject = "/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/data/IEMOCAP_full_release/Session" + str(digit) + "/dialog/transcriptions/"
          
    # for every file in this folder
    for f in os.listdir(ev_subject):
        
        # look at each text file that ends like this
        # if the file extension is atr.txt
        if f[-4:] != ".txt":
            continue
        
        # open the file
        file = open(str(ev_subject + '/' + f))
        
        # Read each line in the file and append to an object
        while True:
            
            # while there's a line to read, read it
            this_line = file.readline()
            
            # if we've reached the end of the file
            if not this_line:
                break
            
            # append the line to the list
            transcriptions.append(this_line)
    
    
transcription_text = []
transcription_tags = []

# Split data by :
# Step through the list of labels
for i in range(len(transcriptions)):
    # split the instance by :
    splits = transcriptions[i].split(":")
    # append to a list
    tags = splits[0]
    tags = tags.split(" ")
    tags = tags[0]
    text = splits[1]
    text = text[1:]
    transcription_tags.append(tags)
    transcription_text.append(text)

transcriptions_df = pd.DataFrame([transcription_tags, transcription_text]).T
transcriptions_df.columns = ['transcription_tags', 'transcription_text']
transcriptions_df.index = transcriptions_df['transcription_tags']

# Amalgamate transcriptions with VAD means and wav tags
# Join tags of unique wavs to VAD averages
# for some reason pd.concat was returning nans for the text
text_get = []

for j in labels_mean_wavs.index:
    the_text = transcriptions_df['transcription_text'].loc[j]
    text_get.append(the_text)

labels_mean_wavs['transcription'] = text_get

with open('/Volumes/COLESLAW_1TB/BELL_LABS/unique_wavs_VAD_transcriptions.data',
          'wb') as new_data:
    # store the data as binary data stream - protocol 2 is so it can be used on Kevin Street
    pickle.dump(labels_mean_wavs, new_data, protocol=4)
    print('Saved labels')

