#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:26:54 2020

@author: billcoleman

From: https://towardsdatascience.com/develop-a-nlp-model-in-python-deploy-it-with-flask-step-by-step-744f3bdd7776
"""

from flask import Flask, render_template, url_for, request, redirect
import pandas as pd
import numpy as np
import re
# to load vectorizer, models
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import speech_recognition as sr

# Audio functions
import librosa as lib
# Normalise data
from sklearn.preprocessing import MinMaxScaler

# To load cnn model
# import tensorflow as tf
# =============================================================================
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# =============================================================================
# from keras.models import model_from_json
import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.initializers import glorot_uniform

r = sr.Recognizer()

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

def clean_lemma(text):
    
    corpus = []
    
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    return corpus


def strip_silence(file_to_strip):
    '''
    Takes a non-treated file and strips silent segments from it - provides an
    array as final output
    '''
    intervals = lib.effects.split(file_to_strip,
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


def convert_to_lpms(raw_silenced):
    '''
    Take a raw wave with the silence removed and convert to a LPMS matrix
    '''
    log_pow = lib.feature.melspectrogram(y=raw_silenced,
                                             sr=16000,
                                             n_mels=40, 
                                             win_length=1024,
                                             hop_length=256)
    
    # scale to dB
    log_pow_db = lib.power_to_db(log_pow, ref=np.max)
    
    return log_pow_db


def chunk_scale_lpms_matrix(lpms_matrix, data_list):
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
        
        # increment start and end of chunks
        track_start += 1
        track_end += 1


# to pick up wav
@app.route('/upload_sound', methods=['POST'])
def upload_sound(text=None):
    
    # Open file and write binary (blob) data
    f = open('./file.wav', 'wb')
    f.write(request.data)  # was getfile = f.write(request.data) 
    f.close()
    
    return ".wav Stored!"


# =============================================================================
# @app.route('/messages', methods = ['POST'])
# def api_message():
#     f = open('./file.wav', 'wb')
#     f.write(request.data)
#     f.close()
#     return "Binary message written!"
# =============================================================================

@app.route('/result', methods=['GET'])
def result():
    
    '''
    Load model and vectorizer
    '''
    print("JUMP TO PREDICT!!!")
    
    getfile = "file.wav"
    juice = sr.AudioFile(getfile)
    with juice as source:
        audio = r.record(source)
        text = r.recognize_google(audio)
        print("TRANSCRIPTION IS: ", text)
    
    # load VAD text models
    model_val = joblib.load('model_text_valence_iemocap.pkl')
    model_act = joblib.load('model_text_activation_iemocap.pkl')
    model_dom = joblib.load('model_text_dominance_iemocap.pkl')
    vect_file = joblib.load('vect_obj_iemocap.pkl')
    
    # munge text
    message = clean_lemma(text)
    message = vect_file.transform(message).toarray()

    # Text predictions
    predictions_V = model_val.predict(message)
    predictions_A = model_act.predict(message)
    predictions_D = model_dom.predict(message)
      
    # trigger functions to read wav, munge it and predict VAD from audio
    
    # List to store lpms matrices
    wav_samples = []
    
    # get datapoints and sample rate of file and load it
    samples, sar = lib.load(getfile, sr=None)
    silence_stripped = strip_silence(samples)
    lpms_ified = convert_to_lpms(silence_stripped)
    chunk_scale_lpms_matrix(lpms_ified, wav_samples)

    # model_audio = keras.models.load_model('model_audio_iemocap_v2.h5')
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model_audio = load_model('model_audio_iemocap_v2.h5')

    print("Loaded model from disk")
    
    # As wav_samples is a list I can't use array indexing on it
    # convert to ndarray
    wav_samples = np.array(wav_samples)
    print("wav_samples length: ", len(wav_samples))
    print("wav_samples type: ", type(wav_samples))
    
    nRows, nCols, nDims = 20, 25, 1
    wav_samples = wav_samples.reshape(wav_samples.shape[0], nRows, nCols, nDims)
    print("RESHAPED wav_samples: ", wav_samples.shape)
    
    # Step through each 0.4 sec chunk and make a prediction, store it    
    audio_predictions = model_audio.predict(wav_samples, batch_size=32, verbose=2)
    
    print("Predictions list length: ", len(audio_predictions))
    print("Predictions slot[0] length: ", len(audio_predictions[0]))
    
    # Calculate the mean of each prediction
    audio_pred_val = audio_predictions[:, 0].mean()
    audio_pred_act = audio_predictions[:, 1].mean()
    audio_pred_dom = audio_predictions[:, 2].mean()
    
    print("Length of frame data: ", len(audio.frame_data))
    print("File sample_rate: ", audio.sample_rate)
    print(predictions_V, audio_pred_val)
    print(predictions_A, audio_pred_act)
    print(predictions_D, audio_pred_dom)
    
    text_ = [str(text)]
    
    # Provide predictions to results page
    return render_template('result.html',  # was result.html
                           pred_words=text_,
                           pred_V=predictions_V,
                           pred_A=predictions_A,
                           pred_D=predictions_D,
                           pred_Vaud=audio_pred_val,
                           pred_Aaud=audio_pred_act,
                           pred_Daud=audio_pred_dom)

if __name__ == '__main__':
	app.run(debug=True, threaded=False)