# Name: Brady Hoeksema
# Email: brady.hoeksema@uleth.ca
# S_ID: 001221045
# Class: CPSC 5310 (NLP)
# Assignment 3

# Imports
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
nltk.download("stopwords")

df = pd.read_csv('news.csv')

# Split data for testing/training purposes
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

# Lemmatization function
def lemma_stem_text(words_list):
    # Lemmatizer
    text = [lemmatizer.lemmatize(token.lower()) for token in words_list]
    text = [lemmatizer.lemmatize(token.lower(), "v") for token in text]
    return text

re_negation = re.compile("n't ")

# Function to transform abbreviated negations to standard form
def negation_abbreviated_to_standard(sent):
    sent = re_negation.sub(" not ", sent)
    return sent

def review_to_words(text):
    # 2. Transform abbreviated negations to the standard form.
    standard_text = negation_abbreviated_to_standard(text)

    # 3. Remove non-letters and non-numbers
    letters_numbers_only = re.sub("[^a-zA-Z_0-9]", " ", standard_text)

    # 4. Convert to lower case and split into individual words (tokenization)
    
    words = letters_numbers_only.lower()
    words = word_tokenize(words)

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
vectorizer = TfidfVectorizer(max_features = 10000, ngram_range = (1, 2))

#Create training set with the words encoded as features of the reviews
train_data_features = vectorizer.fit_transform(cleaned_train_reviews)

#Define the model
model = LogisticRegression(random_state = 0, solver = 'lbfgs' , multi_class = 'multinomial')

#Train the model
model.fit(train_data_features, y_train)

clean_test_reviews = []
for i in X_test:
    clean_test_reviews.append(review_to_words(i))

test_data_features = vectorizer.transform(clean_test_reviews)

result = model.predict(test_data_features)
accuracy = accuracy_score(y_test, result)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, result))
