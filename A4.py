# Name: Brady Hoeksema
# Email: brady.hoeksema@uleth.ca
# S_ID: 001221045
# Class: CPSC 5310 (NLP)
# Assignment 4

# Imports
import pandas as pd
import numpy as np

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('news.csv')

# Split data for testing/training purposes
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

stopwords_list = set(stopwords.words("english"))
# We remove negation words in list of stopwords
no_stopwords = ["not","don't",'aren','don','ain',"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
               'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
               "won't", 'wouldn', "wouldn't"]

for no_stopword in no_stopwords:
    stopwords_list.remove(no_stopword)

lemmatizer = WordNetLemmatizer()

# Lemmatization function
def lemma_stem_text(words_list):
    # Lemmatizer
    text = [lemmatizer.lemmatize(token.lower()) for token in words_list]
    text = [lemmatizer.lemmatize(token.lower(), "v") for token in text]
    return text

def review_to_words(text):
    # Remove non-letters and non-numbers
    letters_numbers_only = re.sub("[^a-zA-Z_0-9]", " ", text)

    # Convert to lower case and split into individual words (tokenization)
    
    words = letters_numbers_only.lower()
    words = word_tokenize(words)

    # Remove stop words
    meaningful_words = [w for w in words if not w in stopwords_list]

    # Apply lemmatization function
    lemma_words = lemma_stem_text(meaningful_words)

    # Join the words back into one string separated by space, and return the result.
    return( " ".join(lemma_words))

# We initialize an empty list to add the clean reviews
cleaned_train_reviews = []

# We loop over each review and clean it
for i in X_train:
    cleaned_train_reviews.append(review_to_words(i))

# Word2Vec Hyperparameters
embeddingsSize=100
model=Word2Vec(X_train, window=5, min_count=1, workers=4)

# Function to get the vectors for Word2Vec
def getVectors(dataset):
  singleDataItemEmbedding=np.zeros(embeddingsSize)
  vectors=[]
  for dataItem in dataset:
    wordCount=0
    for word in dataItem:
      if word in model.wv.index_to_key:
        singleDataItemEmbedding=singleDataItemEmbedding+model.wv[word]
        wordCount=wordCount+1
  
    singleDataItemEmbedding=singleDataItemEmbedding/wordCount  
    vectors.append(singleDataItemEmbedding)
  return vectors

trainReviewVectors=getVectors(X_train)
testReviewVectors=getVectors(X_test)

# Neural network
from sklearn.neural_network import MLPClassifier

clfMLP = MLPClassifier(hidden_layer_sizes=(100,10))
clfMLP.fit(trainReviewVectors, y_train)
  
testLabelsPredicted=list(clfMLP.predict(testReviewVectors))

# Testing the accuracy of the model
accuracy = accuracy_score(y_test, testLabelsPredicted)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, testLabelsPredicted))