#Imports
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
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
stop_tokens = []
stop_words = set(stopwords.words("english"))
remove_punc = tokens
#Removing punctuation
remove_punc = remove_punc.translate(str.maketrans('','',string.punctuation))
filtered = [word for word in remove_punc if word.lower() not in stop_words]
stop_tokens.append(filtered)

print("Part b.) Without Stopwords")