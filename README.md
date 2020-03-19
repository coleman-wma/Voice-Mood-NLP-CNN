# NLP-Emotion_Voice
Repo for Dublin.AI personal project looking at developing a voice biometric monitoring solution for mental health applications.

# Title: NLP/Emotion prediction

# Problem:
The voice is the ‘window to the soul’ but there are no apps which use the voice to give users insight to their current state-of-mind and track this month-on-month.
# Similar to:
Garmin/Fitbit/Oura Ring - these monitor fitness activity or other physiological measures (heart rate, heart rate variability, body temperature) and provide statistical analysis of current health state. Also, there are a number of mood apps which allow people to monitor their mood and advise on methods to manage patterns. Important for people who are bipolar, have anxiety or are managing depression, but also for those interested in their mental wellness.
# Objective:
Provide health monitoring using sentiment and acoustic analysis of voice signals predicting current state-of-mind and emotion.
Build an MVP (minimal viable product) using machine learning to (1) predict user sentiment and (2) predict user emotion.

# Milestones:
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

# Deliverables:
* Laptop prototype that takes voice input and provides measures of sentiment and emotion
    * Begin with selecting an instance from the test set - finds audio, finds transcription, makes prediction
    * 2nd version - takes voice audio recording as input - does voice to text, does prediction
* Web deployment

# Assets:
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
* IEMOCAP dataset: https://sail.usc.edu/iemocap/
* Kaggle emotion in text competition: https://www.kaggle.com/c/sa-emotions/data

# Tech Specs:
* Python

# Risks:
* Time constraints
* Complexity - implementation of other code base?
* Demonstrability
