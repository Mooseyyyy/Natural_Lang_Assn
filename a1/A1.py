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

#Part A
print("Part a.) With Stopwords")
print("The number of word tokens in the corpus is: ", len(tokens))
print("The number of unique words are: ",unique(tokens))

# Part B
stop_tokens = []
stop_words = set(stopwords.words("english"))
temp = tokens
# Removing punctuation
def remove_punctuation(not_removed):
    trans = str.maketrans('', '', string.punctuation)
    stripped = [x.translate(trans) for x in not_removed]
    return stripped

remove_punc = [remove_punctuation(x) for x in temp]

for x in remove_punc:
    if x.lower() not in stop_words:
        stop_tokens.append(x)

print("Part b.) Without Stopwords")
print("The number of word tokens in the courpus is: ", len(stop_tokens))
print("The number of unique words are: ", unique(stop_tokens))

# Part C
lem = WordNetLemmatizer()
stoplem_tokens = []
for x in stop_tokens:
    stoplem_tokens.append(lem.lemmatize(x))

print("Part c.) Without Stopwords and Lemmatization")
print("The number of word tokens in the corpus is: ", len(stoplem_tokens))
print("The number of unique words are: ", unique(stoplem_tokens))

# Part D
ps = PorterStemmer()
stopstem_tokens = []
for x in stop_tokens:
    stopstem_tokens.append(ps.stem(x))
print("Part d.) Without Stopwords and Stemming")
print("The number of word tokens in the corpus is: ", len(stopstem_tokens))
print("The number of unique words are: ", unique(stopstem_tokens))
