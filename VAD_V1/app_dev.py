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

r = sr.Recognizer()

# getfile = "/Users/billcoleman/NOTEBOOKS/DublinAI/nlp_emotion/VAD/wav/drummonds_16k.wav"

# =============================================================================
# drummond = sr.AudioFile(getfile)
# with drummond as source:
#     audio = r.record(source)
# 
# r.recognize_google(audio)
# =============================================================================

# =============================================================================
# import pickle
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
# =============================================================================

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

# to pick up wav
@app.route('/upload_sound', methods = ['POST'])
def api_message():
      # Open file and write binary (blob) data
      f = open('./file.wav', 'wb')
      getfile = f.write(request.data)
      f.close()
      
      getfile = "file.wav"
      juice = sr.AudioFile(getfile)
      with juice as source:
          audio = r.record(source)

      text = r.recognize_google(audio)
      print(text)
      # trigger functions to predict VAD from text using 'text' as input
      
      # trigger functions to read wav, munge it and predict VAD from audio
      
      print("Length of frame data: ", len(audio.frame_data))
      print("File sample_rate: ", audio.sample_rate)
      
      return "Binary message written!"

# message = "This is a message to test the comings and the goings"

# =============================================================================
# @app.route('/messages', methods = ['POST'])
# def api_message():
#     f = open('./file.wav', 'wb')
#     f.write(request.data)
#     f.close()
#     return "Binary message written!"
# =============================================================================

@app.route('/result', methods=['POST'])
def predict(prediction=None):
    
    '''
    Load model and vectorizer
    '''
    
    model_val = joblib.load('model_text_valence.pkl')
    model_act = joblib.load('model_text_activation.pkl')
    model_dom = joblib.load('model_text_dominance.pkl')
    vect_file = joblib.load('vect_obj.pkl')

    if request.method == 'POST':
        message = request.form['message']
        message = clean_lemma(message)  # [message]

        message = vect_file.transform(message).toarray()
        
        reg_V = model_val
        reg_A = model_act
        reg_D = model_dom
        predictions_V = reg_V.predict(message)
        predictions_A = reg_A.predict(message)
        predictions_D = reg_D.predict(message)
        
        return render_template('result.html',
                               pred_V=predictions_V,
                               pred_A=predictions_A,
                               pred_D=predictions_D)

if __name__ == '__main__':
	app.run(debug=True)