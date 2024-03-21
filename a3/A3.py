# Name: Brady Hoeksema
# Email: brady.hoeksema@uleth.ca
# S_ID: 001221045
# Class: CPSC 5310 (NLP)
# Assignment 3

# Imports
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
nltk.download("stopwords")

df = pd.read_csv('news.csv')

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

stopwords_list = set(stopwords.words("english"))
# We remove negation words in list of stopwords
no_stopwords = ["not","don't",'aren','don','ain',"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
               'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
               "won't", 'wouldn', "wouldn't"]

for no_stopword in no_stopwords:
    stopwords_list.remove(no_stopword)

lemmatizer = WordNetLemmatizer()

# function that receive a list of words and do lemmatization:
def lemma_stem_text(words_list):
    # Lemmatizer
    text = [lemmatizer.lemmatize(token.lower()) for token in words_list]
    text = [lemmatizer.lemmatize(token.lower(), "v") for token in text]
    return text

import re
re_negation = re.compile("n't ")
# function that receive a sequence of woords and return the same sequence transforming
# abbreviated negations to the standard form.
def negation_abbreviated_to_standard(sent):
    sent = re_negation.sub(" not ", sent)
    return sent

# We get the text of reviews in the training set
reviews = X_train
print(type(reviews))
def review_to_words(raw_review):
    # 2. Transform abbreviated negations to the standard form.
    review_text = negation_abbreviated_to_standard(raw_review)

    # 3. Remove non-letters and non-numbers
    letters_numbers_only = re.sub("[^a-zA-Z_0-9]", " ", review_text)

    # 4. Convert to lower case and split into individual words (tokenization)
    words = np.char.lower(letters_numbers_only.split())

    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stopwords_list]

    # 6. Apply lemmatization function
    lemma_words = lemma_stem_text(meaningful_words)

    # 7. Join the words back into one string separated by space, and return the result.
    return( " ".join(lemma_words))

# We initialize an empty list to add the clean reviews
cleaned_train_reviews = []

# We loop over each review and clean it
for i in X_train:
    cleaned_train_reviews.append(review_to_words(i))

#Hyperparameters
vectorizer = TfidfVectorizer(max_features = 20000, ngram_range = (1, 2))

#Create training set with the words encoded as features of the reviews
train_data_features = vectorizer.fit_transform(cleaned_train_reviews)

#Define the model
model = LogisticRegression(random_state = 0, solver = 'lbfgs' , multi_class = 'multinomial')

#Train the model
model.fit(train_data_features, y_train)

num_reviews = len(X_test)
clean_test_reviews = []

for i in range(0, num_reviews):
    clean_review = review_to_words(X_test[i])
    clean_test_reviews.append(clean_review)

test_data_features = vectorizer.transform(clean_test_reviews)

result = model.predict(test_data_features)
