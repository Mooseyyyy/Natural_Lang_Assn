#Imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import csv
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Read in the csv files
reader = csv.reader(open('news.csv'))

# Tokenize all words in file
tokens = []
for line in reader:
    for field in line:
        tokens.append(word_tokenize(field))

# Function to pull out unique words
def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    print(len(unique_list))

print(tokens[:6])
