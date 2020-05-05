# NLP-Emotion_Voice
Repo for Dublin.AI personal project looking at developing a voice biometric monitoring solution for mental health applications.

[Click here to play a Loom video explaining the project.](https://www.loom.com/share/42298b03111c4ed6a7d4d224b65999b5)

Above you'll see two Jupyter Notebooks which outline data preparation ('Emotion_NLP_Audio_data_preparation.ipynb') and basic model training ('Emotion_NLP_models.ipynb').

There are a number of theories of emotion. Here, I'm going to use the dimensional model, which differentiates emotions on a number of axes: Valence (pleasant/unpleasant), Activation (high energy/low energy) and Dominance (dominant/submissive).

![Emotional Dimensions](https://github.com/coleman-wma/Voice-Mood-NLP-CNN/blob/master/images/dimensions_of_emotion.png)

There are a number of datasets which provide stimuli with scores on these dimensions. Here I'm using [IEMOCAP](https://sail.usc.edu/iemocap/).

Further code for building models is in the 'python' folder.

Code for the web application is in the 'web_app' folder.

This application creates a front end where users can record a snippet of themselves talking. This is converted to text using Google's speech-to-text, and I then predict emotion on three dimensions (valence, activation and dominance) using both the audio and text outputs.

This frames emotion prediction as a regression task. I apply 3 NLP models to the text output and a CNN to the audio output.

# Scoping Document

## Problem:
The voice is the ‘window to the soul’ but there are no apps which use the voice to give users insight to their current state-of-mind and track this month-on-month.
## Similar to:
Garmin/Fitbit/Oura Ring - these monitor fitness activity or other physiological measures (heart rate, heart rate variability, body temperature) and provide statistical analysis of current health state. Also, there are a number of mood apps which allow people to monitor their mood and advise on methods to manage patterns. Important for people who are bipolar, have anxiety or are managing depression, but also for those interested in their mental wellness.
## Objective:
Provide health monitoring using sentiment and acoustic analysis of voice signals predicting current state-of-mind and emotion.
Build an MVP (minimal viable product) using machine learning to (1) predict user sentiment and (2) predict user emotion.

## Milestones:
* Identify appropriate dataset(s) to predict sentiment from text
    * Or is there a pre-trained model that could be used?
* Identify appropriate dataset(s) to predict emotion from speech
* Identify pre-trained model for converting speech to text
* Build working prototype on laptop and assess performance
* Model refinement if feasible
    * Technical consideration
    * Available processing resources
* Web deployment with suitable visuals
* Simulation of a dashboard which provides overview of historic data (subject to time constraints)

## Deliverables:
* Laptop prototype that takes voice input and provides measures of sentiment and emotion
    * Begin with selecting an instance from the test set - finds audio, finds transcription, makes prediction
    * 2nd version - takes voice audio recording as input - does voice to text, does prediction
* Web deployment

## Assets:
* Data: Kaggle & other publicly available datasets for NLP
* Data: Emotion in speech datasets collated for Bell Labs
* Model for speech to text conversion - Google, Mozilla??
* Knowledge Resources: towardsdatascience.com, NLTK, GloVe
* Garmin Dashboard: https://connect.garmin.com/
* Fitbit: https://www.fitbit.com/ie/app
* Oura Ring: https://ouraring.com/why-oura
* Mood apps: https://www.everydayhealth.com/columns/therese-borchard-sanity-break/the-6-best-mood-apps/
* Simple python implementation of NLP/sentiment analysis: https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk
* Links to a number of datasets and structure of analysis/code: https://towardsdatascience.com/building-a-vocal-emotion-sensor-with-deep-learning-bedd3de8a4a9
* Sentiment analysis, some datasets and code: https://www.kdnuggets.com/2018/08/emotion-sentiment-analysis-practitioners-guide-nlp-5.html
* Emotion prediction using audio and text inputs, SOTA: https://github.com/david-yoon/multimodal-speech-emotion
* IEMOCAP dataset: https://sail.usc.edu/iemocap/ - 3 to 5 day request time
* Kaggle emotion in text competition: https://www.kaggle.com/c/sa-emotions/data
* JULIE EmoBank dataset: https://github.com/JULIELab/EmoBank - this one scores on Valence, Activation and Dominance scales.
* RECOLA - audio - also scores on Valence, Activation and Dominance - requires request from Professor - 
* USC Creative IT - audio - scores on VAD - https://sail.usc.edu/CreativeIT/ImprovRelease.htm - 3 to 5 day request time

## Tech Stack:
* Python
* Flask, HTML, Javascript, CSS

## Risks:
* Time constraints
* Complexity - implementation of other code base?
* Demonstrability
* Datasets - the more interesting ones are not freely available for download - they must be requested, which takes 3 to 5 working days to approve
