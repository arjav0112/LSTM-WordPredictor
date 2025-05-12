from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import numpy as np
import pandas as pd
import streamlit as st
import pickle

with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the model
model = load_model('hamlet_model.h5')

def predict_sentiment(model,text,tokenizer,next_words,max_sequnce_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) > max_sequnce_length:
            token_list = token_list[-max_sequnce_length:]
        
        token_list = pad_sequences([token_list],maxlen=max_sequnce_length-1,padding='pre')

        predicted_word = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted_word, axis=-1)
        output_word = ""

        for word,index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
                
        text += " " + output_word
    return text

st.title("Next Words Prediction")
st.write("This app predicts the next words in a sentence using a pre-trained LSTM model.")

st.write("Number of you want to predict: ")
next_words = st.number_input("Next Words", min_value=1, max_value=30)

st.write("Enter a sentence below and click the button to predict the next words.")
input_text = st.text_area("Input Text", "Type your text here...")
try:
    if st.button("Predict Next Words"):
        if input_text:
            # next_words = 5
            max_sequence_length = model.input_shape[1] + 1
            predicted_text = predict_sentiment(model, input_text, tokenizer, next_words, max_sequence_length)
            st.write("Predicted Text: ", predicted_text)
        else:
            st.write("Please enter some text to predict the next words.")
except Exception as e:
    st.write("An error occurred: ", str(e))
    st.write("Please check your input and try again.")
