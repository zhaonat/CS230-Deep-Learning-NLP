import nltk
import os
import json
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pickle
from settings import ROOT_DIR
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_extraction.text
from sklearn.feature_extraction.text import TfidfTransformer
from collections import defaultdict
import operator
from time import time
import numpy as np




data_dir = ROOT_DIR+'\\financial_times\\FT-archive-2013\\';
out_dir = ROOT_DIR+'\\processed_data\\';
print(dir)
counter = 0;
raw_document_data = list(); #stores all texts as single string
word_sentence_data = list(); #stores text as individual words found in the article

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, ngram_range=(1,3))

counter = 0;
count_vect = CountVectorizer();
for file in os.listdir(data_dir): #each file is a document in the corpus

    data = json.load(open(data_dir+file,  encoding="utf8"));
    #pprint(data)

    text = data['bodyXML']

    raw_document_data.append(text);
    counter+=1;
    print(counter)
    if(counter> 1000):
        break;

word_counts = count_vect.fit_transform(raw_document_data)
#we can actually convert word_counts to tf-idf
tf_transformer = TfidfTransformer(use_idf=False).fit(raw_document_data)

tfidf_vectorizer.fit_transform(raw_document_data);
print(tfidf_vectorizer.vocabulary_)
print(tfidf_vectorizer.idf_.shape)




