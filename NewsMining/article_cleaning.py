import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
import re
import string
import argparse
import itertools
from unidecode import unidecode
import pandas as pd

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import backend
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Reshape, InputLayer, Lambda
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers


regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


def remove_punctuation(docs):
    docs_no_punctuation = []

    for review in docs:
        new_review = []
        for token in review:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)

        docs_no_punctuation.append(new_review)
    return docs_no_punctuation

def remove_stop_words(docs):
    docs_no_stopwords = []
    for doc in docs:
        new_term_vector = []
        for word in doc:
            if not word in stopwords.words('english'):
                new_term_vector.append(word)
        docs_no_stopwords.append(new_term_vector)
    return docs_no_stopwords

def stemming_and_lemmatizing(docs):
    porter = PorterStemmer()
    snowball = SnowballStemmer('english')
    wordnet = WordNetLemmatizer()

    preprocessed_docs = []

    for doc in docs:
        final_doc = []
        for word in doc:
            final_doc.append(porter.stem(word))
            #final_doc.append(snowball.stem(word))
            #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
        preprocessed_docs.append(final_doc)
    return preprocessed_docs

def read_FT(data_dir):
    counter = 0;
    document_data = list(); #stores all texts as single string

    for f in os.listdir(data_dir):
        data = json.load(open(data_dir+f), encoding="utf8");
        #pprint(data)

        try:
            text = data['bodyXML']
        except KeyError:
            print(f)
            continue
        #clean html from the document while it is still a string
        text = remove_tags(text);
        document_data.append(text);

        counter+=1;
    return document_data

def read_Guardian(data_path):
    document_data = list(); #stores all texts as single string
    section_label = []
    data = pd.read_csv(data_path, index_col=0, encoding='utf-8');

    doc_counter = 0
    num_examples = 100000000
    for article in data.T.columns:
        print(doc_counter)
        if doc_counter >= num_examples:
            break
        text = (data.T[article]['text'])
        if(isinstance(text, str) or isinstance(text, unicode)):
            document_data.append(unidecode(text));
            section_label.append(data.T[article]['sectionId'])
            doc_counter += 1
        else:
            #print(str(text) +', ' +str(no_text_counter));
            pass
    return document_data, section_label

def process_dataset(document_data, remove_punc=False, remove_sw=True, stem=True, sent=True):
    ## tokenization step
    document_data = [word_tokenize(doc) for doc in document_data]

    ## remove punctuation
    if remove_punc:
        document_data = remove_punctuation(document_data)

    ## remove stop words
    if remove_sw:
        document_data = remove_stop_words(document_data)

    ## stemming and lemmatizing
    if stem:
        document_data = stemming_and_lemmatizing(document_data)

    ## convert preprocessed doc back into sentences

    preprocessed_articles = [unidecode(' '.join(x)) for x in document_data];
    if not sent:
        return preprocessed_articles
    preprocessed_sentences = [sent_tokenize(x) for x in preprocessed_articles]
    return preprocessed_sentences

def process_to_array(guardian_corpus, labels):
    max_words = 20  # number of words in a sentence
    max_sentences = 20 # number of sentences in an article

    ### pad/truncate sentences to make sure each article has same number of sentences
    guardian_pad_sent = []
    for article in guardian_corpus:
        if len(article) == max_sentences:
            guardian_pad_sent.append(article)
        elif len(article) > max_sentences:
            guardian_pad_sent.append(article[0:max_sentences])
        else:
            new_article = [' '] * (max_sentences - len(article)) + article
            guardian_pad_sent.append(new_article)

    guardian_corpus = guardian_pad_sent


    ### merge labels with less than 50 samples to be "others"
    labels = [x.encode('UTF8') for x in labels]
    labels_dict = dict((x,labels.count(x)) for x in set(labels))
    others = []
    for label in labels_dict:
        if labels_dict[label] < 50:
            others.append(label)

    new_labels = ['others' if x in others else x for x in labels]
    labels = labels
    num_labels = len(set(labels))


    ### convert labels to one hot encoding
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


    ### tokenize and pad/truncate words to make sure each sentence (not article) has same number of words
    num_word = 1000
    tokenizer = Tokenizer(num_words=num_word)

    data = []

    for article in guardian_corpus:
        tokenizer.fit_on_texts(article)
        sequences = tokenizer.texts_to_sequences(article)

        # pad/truncate words to make sure each sentence has same number of words
        new_article = pad_sequences(sequences, maxlen=max_words, padding='pre', truncating='pre')
        data.append(new_article)


    data = np.stack(data, axis = 0)
    return data, np.asarray(onehot_encoded)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='raw_data/Guardian_10K_epoch_1.csv')
    parser.add_argument('-o', '--output', default='processed_data/Guardian_epoch_1_with_labels')
    parser.add_argument('-p', '--remove_punctuations', action='store_true')
    parser.add_argument('-s', '--remove_stop_words', action='store_true')
    parser.add_argument('-t', '--stem', action='store_true')
    parser.add_argument('--sent', action='store_true', help='set to split sentences')
    parser.add_argument('-d', '--dataset', default='Guardian', help='FT/Guardian')
    args = parser.parse_args()
    if args.dataset == 'FT':
        document_data, labels = read_FT(args.input)
    elif args.dataset == 'Guardian':
        document_data, labels = read_Guardian(args.input)
    docs = process_dataset(document_data, args.remove_punctuations, args.remove_stop_words, args.stem, args.sent)
    docs_array, onehot_encoded = process_to_array(docs, labels)
    np.save('docs', docs_array)
    np.save('labels', onehot_encoded)


    #pickle.dump([docs, labels], open(args.output, 'wb'));
