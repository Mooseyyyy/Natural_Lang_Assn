# Name: Brady Hoeksema
# Email: brady.hoeksema@uleth.ca
# S_ID: 001221045
# Class: CPSC 5310 (NLP)
# Assignment Two

# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('news.csv')

X = df.drop(['label'],axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

