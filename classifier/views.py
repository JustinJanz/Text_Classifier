from django.shortcuts import render
from django.http import HttpResponse
from .forms import ContactForm
from sklearn.datasets import fetch_20newsgroups
import joblib
from django.contrib import messages

import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow import keras

from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np

from .models import Text

from glob import glob
import random
import os
import pickle


model = joblib.load("D:\My Projects\TCS\prediction.pkl")
data = fetch_20newsgroups()


model1 = keras.models.load_model('D:\My Projects\TCS\LSTM_model.h5')


posts = [
    {
        'Text': 'Sample Text',
        'Category': 'sports',
        'Sub_Category': 'Negative'
    }
]
SEQUENCE_LENGTH = 300
oov_token = None
num_words = 10000


tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
data["tokenizer"] = tokenizer

data["int2label"] =  {0: "negative", 1: "positive"}
data["label2int"] = {"negative": 0, "positive": 1}


def classify(request):
    
    text_input = ""
    v =False
    if request.method =='POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            text_input = form.cleaned_data['Text']
            v = True
    x = str(my_predictions(text_input,model))
    y = get_predictions(text_input)
    context = {
        'given_text': text_input,
        'Category':x,
        'Sub_Category':y[1]
    }
    form = ContactForm()
    return render(request,"classifier/home.html",{'form': form, 'category': x,'text':text_input,'successful_submit':v,'posts': context })


def my_predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction1 = model.predict([my_sentence])
    return all_categories_names[prediction1]  


def get_predictions(text):
    sequence = data["tokenizer"].texts_to_sequences([text])
    # pad the sequences
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model1.predict(sequence)[0]
   
    return prediction, data["int2label"][np.argmax(prediction)]










