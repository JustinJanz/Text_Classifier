


import seaborn as sns
import NumPy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set() 

# Load the dataset
data1 = fetch_20newsgroups()
# Get the text categories
text_categories1 = data1.target_names

# Build the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model using the training data
    
train_data1 = fetch_20newsgroups(subset="train", categories=text_categories1)
#define the test set
test_data1 = fetch_20newsgroups(subset="test", categories=text_categories1) 
model.fit(train_data.data, train_data.target)
# Predict the categories of the test data
predicted_categories = model.predict(test_data.data)


def my_predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction1 = model.predict([my_sentence])
    return all_categories_names[prediction1]  

import  joblib
joblib.dump(model,'prediction.pkl')