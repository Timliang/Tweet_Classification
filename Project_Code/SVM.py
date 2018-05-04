# -*- coding: utf-8 -*-
"""
Created on Tue May 01 01:24:26 2018

@author: HP
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 19:47:59 2018

@author: HP
"""

import warnings
import pandas as pd
import nltk
import collections
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import ngrams
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
 


warnings.filterwarnings("ignore", category = DeprecationWarning)
### Download dataset for our model
tweets = pd.read_csv("D:\\Term-X\\ERG 3020\\sailors2017-master\\data\\labeled-data-singlelabels-train.csv")
tweets_test = pd.read_csv("D:\\Term-X\\ERG 3020\\sailors2017-master\\data\\labeled-data-singlelabels-test.csv")

C1_counts = tweets.C1.value_counts()
C2_counts = tweets.C2.value_counts()
number_of_tweets = tweets.id.count()

def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet)   
    tokens = nltk.word_tokenize(only_letters)[2:]
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

def normalize(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ",tweet)
    return only_letters


def ngrams(input_list):
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams+trigrams

def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

def C1_totarget(cag):
    try:
        return {
        'Food': 0,
        'None': 1,
        'Energy' : 2,
        'Water': 3,
        'Medical': 4,
        'need': 5
        }[cag]
    except:
        return 6

def C2_totarget(cag):
    try:
        return{
        'resource': 0,
        'need': 1
        }[cag]
    except:
        return 3

### Eliminate those disturbing tokens using our normalization function      
pd.set_option('display.max_colwidth', -1)
tweets.text = tweets.text.apply(normalize)
tweets_test.text = tweets_test.text.apply(normalize)
tweets['normalized_tweet'] = tweets.text.apply(normalizer)
tweets_test['normalized_tweet'] = tweets_test.text.apply(normalizer)

### Construct bag-of-words use our new ngrams function
tweets['grams'] = tweets.normalized_tweet.apply(ngrams)
tweets_test['grams'] = tweets_test.normalized_tweet.apply(ngrams)

### Linear SVM classifier
# Transform each sentence into a vector which is as long as the list of all words observed in our training data
### The features of 1-gram and 2-gram will be extracted, we can store some local information through this operation
count_vectorizer = CountVectorizer(ngram_range=(1,2)) 
vectorized_data = count_vectorizer.fit_transform(tweets.text)
vectorized_data_test = count_vectorizer.transform(tweets_test.text)
#########
targets_train_1 = tweets.C1.apply(C1_totarget)
targets_train_2 = tweets.C2.apply(C2_totarget)
targets_test_1 = tweets_test.C1.apply(C1_totarget)
targets_test_2 = tweets_test.C2.apply(C2_totarget)
#########
data_train_index_1 = vectorized_data[:,0]
data_train_1 = vectorized_data[:,1:]
data_test_index_1 = vectorized_data_test[:,0]
data_test_1 = vectorized_data_test[:,1:]
### Train our model
clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=1., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train_1, targets_train_1)
### Evaluation of our results
C1_pred = clf.predict(data_test_1)
print(clf.score(data_test_1, targets_test_1))
print(confusion_matrix(targets_test_1, C1_pred))
#########
data_train_index_2 = vectorized_data[:,0]
data_train_2 = vectorized_data[:,1:]
data_test_index_2 = vectorized_data_test[:,0]
data_test_2 = vectorized_data_test[:,1:]

clf = OneVsRestClassifier(svm.SVC(gamma=0.01, C=1., probability=True, class_weight='balanced', kernel='linear'))
clf_output = clf.fit(data_train_2, targets_train_2)
C2_pred = clf.predict(data_test_2)
print(clf.score(data_test_2, targets_test_2))
print(confusion_matrix(targets_test_2, C2_pred))







