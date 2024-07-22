import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

# ignore wannings
import warnings
warnings.filterwarnings('ignore')



word_index = imdb.get_word_index()
reversed_word_index = {v:k for k, v in word_index.items()}
    
model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    prediction=model.predict(preprocessed_input)
    sentiment ='Positive' if prediction[0][0] > 0.6 else 'Negative'
    return sentiment, prediction[0][0]

## streamlit app
st.title('IMDB movie review sentiment analysis')
st.write('Enter a movie review to classify it as positive or negative')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment ='Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'The sentiment of the movie review is: {sentiment}')
    st.write(f'Confidence: {prediction[0][0]*100:.2f}%')
else:
    st.write('Please enter a movie review.')




