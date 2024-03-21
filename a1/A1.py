# Name: Brady Hoeksema
# Email: brady.hoeksema@uleth.ca
# S_ID: 001221045
# Class: CPSC 5310 (NLP)
# Assignment One

# Imports
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
import csv
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Read in the csv file
reader = csv.reader(open('news.csv'))

# Tokenize all words in file
etokens = []
for line in reader:
    for field in line:
        tokens.append(word_tokenize(field))

# Remove all words from nested list and append to master list
new_tokens =[]
def reorder(my_list):
    global new_tokens
    for item in my_list:
        if isinstance(item, list):
           reorder(item)
        else:
            new_tokens.append(item)

# Function to return the length of unique elements from input list
unique_list = []
def count_unique_elements(my_list):
    global unique_list
    for item in my_list:
        # Item is a list
        if isinstance(item, list):
            count_unique_elements(item)
        else:
            # Item is single element
            if item not in unique_list:
                unique_list.append(item)
 
    return len(unique_list)

# Reorder the tokens into one list
reorder(tokens)
# Remove punctuation from the list
new_tokens = [''.join(c for c in s if c not in string.punctuation) for s in new_tokens]
# Remove empty spaces in list from the removed punctuation
new_tokens = [s for s in new_tokens if s]

# Part A ------------------------------------------------------------------------------------------------------------------------------------------------------
# Output
print("Part a.) With Stopwords")
print("The number of word tokens in the corpus is: ")
print(len(new_tokens))
print("The number of unique words are: ")
print(count_unique_elements(new_tokens))
unique_list.clear()

# Part B ------------------------------------------------------------------------------------------------------------------------------------------------------
# Bring in all stop words in the english language
stop_tokens = []
stop_words = set(stopwords.words("english"))

no_stopwords = []
for x in new_tokens:
    if x not in stop_words:
        no_stopwords.append(x)

# Output
print("Part b.) Without Stopwords")
print("The number of word tokens in the corpus is: ")
print(len(no_stopwords))
print("The number of unique words are: ")
print(count_unique_elements(no_stopwords))
unique_list.clear()

# Part C ------------------------------------------------------------------------------------------------------------------------------------------------------
# Create the lemmatizer and add words to new list after being processed
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in no_stopwords]

# Output
print("Part c.) Without Stopwords and Lemmatization")
print("The number of word tokens in the corpus is: ")
print(len(lemmatized_words))
print("The number of unique words are: ")
print(count_unique_elements(lemmatized_words))
unique_list.clear()

# Part D ------------------------------------------------------------------------------------------------------------------------------------------------------
# Create the stemmer and add words to new list after being processed
stemmer = PorterStemmer()
stopstem_tokens = [stemmer.stem(item) for item in no_stopwords]

# Output
print("Part d.) Without Stopwords and Stemming")
print("The number of word tokens in the corpus is: ")
print(len(stopstem_tokens))
print("The number of unique words are: ")
print(count_unique_elements(stopstem_tokens))
unique_list.clear()
