#Imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd 

df = pd.read_csv("./news.csv")

text_tokens = []
count = 0
for text in df['text']:
    words_tokens = word_tokenize(text)
    text_tokens.append(words_tokens)

print(len(text_tokens))