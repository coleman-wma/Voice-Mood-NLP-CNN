#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:23:12 2020

@author: billcoleman
"""

import pandas as pd
eb = pd.read_csv("data/emobank.csv", index_col=0)

print(eb.shape)
test = eb.head()

# NLP module
import nltk
# https://www.guru99.com/tokenize-words-sentences-nltk.html
from nltk.tokenize import word_tokenize # word tokenize
# from nltk.tokenize import sent_tokenize # sentence tokenize

'''
# Tweets example https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
'''
# To tokenise text - this is a pretrained model
# Don't think I need this
# nltk.download('punkt')

# text normalisation (converting a word to its canonical form - ran, runs, 
# running are forms of 'run')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer

all_text = eb['text']
sample_text = all_text[0:100]

text_tokens = []

# =============================================================================
# def lose_punctuation(zap_this):
#     token = re.sub(r"[,@\'?\.$%_:\"-]", "", zap_this, flags=re.I)
#     return token
# =============================================================================

# ’“”
def lose_punctuation(zap_this):
    token = re.sub(r'[,@\'?\.$%_:\-"’“”]', "", zap_this, flags=re.I)
    return token

# print(all_text[0].translate(str.maketrans('', '', string.punctuation)))
print(lose_punctuation(all_text[0]))

'''
Tokenize text
'''

for i in range(0, len(all_text)):
    tokenized = word_tokenize(all_text[i])
    text_tokens.append(tokenized)

'''
Normalise text
# From: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
'''

# tags as noun, verb whatever...
print(pos_tag(text_tokens[0]))

def lemmatize_sentence(tokens):
    '''
    The function lemmatize_sentence first gets the position tag of each token
    of a tweet. Within the if statement, if the tag starts with NN, the token
    is assigned as a noun. Similarly, if the tag starts with VB, the token is
    assigned as a verb.
    '''
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

print(lemmatize_sentence(text_tokens[0]))


'''
Removing noise from the data
Hyperlinks, twitter handles, punctuation
'''

import re, string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        token = re.sub(r'[,@\'?\.$%_:\-"’“”]', "", token, flags=re.I)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
        # else:
            # print(token)
    return cleaned_tokens

print(remove_noise(text_tokens[0], stop_words))

'''
Applying normalisation and cleaning to all text, store in object
'''

cleaned_text_tokens = []

for tokens in text_tokens:
    cleaned_text_tokens.append(remove_noise(tokens, stop_words))

# compare before with after
print(text_tokens[2510])
print(cleaned_text_tokens[2510])

'''
Determining Word Density
'''

# not sensible to do this for one entry, so lets look at all...

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words = get_all_words(cleaned_text_tokens)

from nltk import FreqDist

freq_dist_pos = FreqDist(all_pos_words)
print(freq_dist_pos.most_common(10))

'''
Preparing data for the model
'''

# =============================================================================
# def get_text_for_model(cleaned_tokens_list):
#     '''
#     convert the text from a list of cleaned tokens to dictionaries with keys as
#     the tokens and True as values
#     '''
#     for tweet_tokens in cleaned_tokens_list:
#         yield dict([token, True] for token in tweet_tokens)
# 
# tokens_for_model = get_text_for_model(cleaned_text_tokens)
# =============================================================================

data_valence = []
data_activation = []
data_dominance = []
data_multi = []

def get_text_for_model_tgt(cleaned_tokens_list):
    '''
    convert the text from a list of cleaned tokens to dictionaries with keys as
    the tokens and True as values
    Change the list to append to different lists
    Change the letter in eb['D'][i] to append different values
    '''
    for i in range(0, len(cleaned_text_tokens)):
        thing = (dict([token, True] for token in cleaned_text_tokens[i]),
                                 eb['V'][i])
        data_valence.append(thing)

get_text_for_model_tgt(cleaned_text_tokens)


# =============================================================================
# positive_dataset = [(tweet_dict, "Positive")
#                      for tweet_dict in tokens_for_model]
# =============================================================================
    
'''
Train and Test Sets
'''

train_data = data_valence[:8000]
test_data = data_valence[8000:]

from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.classify.api import ClassifierI, MultiClassifierI

classifier = NaiveBayesClassifier.train(train_data)

print("Accuracy is:", classify.accuracy(classifier, test_data))

print(classifier.show_most_informative_features(10))

dist = classifier.prob_classify(test_data[1])
for label in dist.samples():
    print("%s: %f" % (label, dist.prob(label)))

examine_probs = []

list(dist.samples())


# Visualisations

import numpy as np
import matplotlib.pyplot as plt

val_objects = np.unique(eb['V'])
val_y_pos = np.arange(len(val_objects))
val_counts = np.unique(eb['V'], return_counts=True)

plt.figure(figsize=(12,6))
plt.bar(val_y_pos, val_counts[1], align='center', alpha=0.5)
plt.xticks(val_y_pos, val_counts[0], rotation=90)
plt.ylabel('Count')
plt.title('Valence Distribution')
plt.show()


act_objects = np.unique(eb['A'])
act_y_pos = np.arange(len(act_objects))
act_counts = np.unique(eb['A'], return_counts=True)

plt.figure(figsize=(12,6))
plt.bar(act_y_pos, act_counts[1], align='center', alpha=0.5)
plt.xticks(act_y_pos, act_counts[0], rotation=90)
plt.ylabel('Count')
plt.title('Activation Distribution')
plt.show()


dom_objects = np.unique(eb['D'])
dom_y_pos = np.arange(len(dom_objects))
dom_counts = np.unique(eb['D'], return_counts=True)

plt.figure(figsize=(12,6))
plt.bar(dom_y_pos, dom_counts[1], align='center', alpha=0.5)
plt.xticks(dom_y_pos, dom_counts[0], rotation=90)
plt.ylabel('Count')
plt.title('Dominance Distribution')
plt.show()

eb['text'][2]
