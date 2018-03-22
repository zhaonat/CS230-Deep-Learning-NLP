
# author - Richard Liao
# Dec 26 2016
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import re
import sys
import os
import settings

os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, CuDNNLSTM, CuDNNGRU
from keras.models import Model
from economist_advanced_class.Attention_Layer import *
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SENT_LENGTH = 200
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

from nltk import tokenize

reviews = []
labels = []
texts = []

## GET DATA
GLOVE_DIR = settings.ROOT_DIR+'\\word_embeddings\\glove.6B\\' #when feeding word embeddings, don't lemmatize
corpus_dir = settings.ROOT_DIR + '\\Guardian_corpus\\'
dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Guardian_corpus_sentences.p';
file2 = 'condensed_label_guardian_dataset.p';
file2 = 'Guardian_corpus_raw.p';
#file = 'Econ_corpus_raw_10k.p'

[econ_docs_as_sentences, not_important] = pickle.load(open(corpus_dir + file, 'rb'));
[econ_corpus, labels] = pickle.load(open(corpus_dir + file2, 'rb'));

# sample = np.random.randint(0, len(labels), 20000);
# econ_corpus = [econ_corpus[i] for i in sample];
# econ_text = [econ_text[i] for i in sample];
# labels = [labels[i] for i in sample];

#convert labels
label_dict = dict();
c = 1;
label_to_num = dict();
for label in labels:
    if(label not in label_dict.keys()):
        label_dict[label] = 1;
        label_to_num[label]= c;
        c+=1;
    label_dict[label]+=1;
labels_index = label_to_num;
numeric_labels = list();
for label in labels:
    numeric_labels.append(label_to_num[label]);
labels = numeric_labels;
num_labels = len(set(labels));

texts = econ_corpus;

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(econ_corpus)
print('tokenization complete')

## we will do some interesting trimming of max sentences and the sentence length of each data point
data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(econ_docs_as_sentences):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    data[i, j, k] = tokenizer.word_index[word]
                    k = k + 1

print('sentence procesing complete')

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical((labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


## USE GLOVE WORD EMBEDDING
print('adding a word embedding')
embeddings_index = {}
f = open(GLOVE_DIR+ 'glove.6B.200d.txt', 'r', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(CuDNNLSTM(512))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

#decoder
review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
## encoding
l_att_sent = AttentionDecoder(32, 16)(review_encoder)


l_lstm_sent = Bidirectional(CuDNNLSTM(256))(l_att_sent)
preds = Dense(num_labels+1, activation='softmax')(l_lstm_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=25, batch_size=32)
print(model.summary())
model.save('hiearchical_LSTM.h5')