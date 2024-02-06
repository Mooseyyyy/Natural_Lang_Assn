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

def count_total_elements(my_list):
    total_elements = 0
 
    for item in my_list:
        if isinstance(item, list):
            total_elements += count_total_elements(item)
        else:
            total_elements += 1

    return total_elements

unique_list = []
def count_unique_elements(my_list):
    global unique_list
    for item in my_list:
        #If item is a list
        if isinstance(item, list):
            count_unique_elements(item)
        else:
            #item is single element
            if item not in unique_list:
                unique_list.append(item)
 
    return len(unique_list)

#Part A
print("Part a.) With Stopwords")
print("The number of word tokens in the corpus is: ")
print(count_total_elements(tokens))
print("The number of unique words are: ")
print(count_unique_elements(tokens))