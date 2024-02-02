#Imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import csv

reader = csv.reader(open('news.csv'))

tokens = []
for line in reader:
    for field in line:
        tokens.append(word_tokenize(field))

def unique(list1): 
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    print(len(unique_list))

#Part A
print("Part a.) With Stopwords")
print("The number of word tokens in the corpus is: ", len(tokens))
print("The number of unique words are: ", unique(tokens))

#Part B
stop_words = set(stopwords.words("english"))
stop_tokens = tokens

print("Part b.) Without Stopwords")