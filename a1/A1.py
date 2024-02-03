#Imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import csv
import pandas as pd
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv('news.csv', encoding='iso-8859-1')

tokens = []
#Cycles through each review and tokenizes everything by word
for word in df['title']:
    word_tokens = word_tokenize(word)
    tokens.append(word_tokens)

print(tokens[:5])
