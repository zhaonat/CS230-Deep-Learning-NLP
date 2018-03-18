
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
from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, regularizers
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, CuDNNLSTM
from keras.models import Model, Sequential

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
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
econ_dir = settings.ROOT_DIR+'\\economist_corpus\\'
dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_sentences.p'
file2 = 'Econ_corpus_no_process.p'
#file = 'Econ_corpus_raw_10k.p'

[econ_corpus, labels] = pickle.load(open(econ_dir + file, 'rb'));
[econ_text, labels] = pickle.load(open(econ_dir + file2, 'rb'));

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
tokenizer.fit_on_texts(econ_text)
print('tokenization complete')

## we will do some interesting trimming of max sentences and the sentence length of each data point
# data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
#
# for i, sentences in enumerate(econ_corpus):
#     for j, sent in enumerate(sentences):
#         if j < MAX_SENTS:
#             wordTokens = text_to_word_sequence(sent)
#             k = 0
#             for _, word in enumerate(wordTokens):
#                 if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
#                     data[i, j, k] = tokenizer.word_index[word]
#                     k = k + 1

data = econ_text;

word_index = tokenizer.word_index
print('Total %s unique tokens.' % len(word_index))
sequences = tokenizer.texts_to_sequences(data)

labels = to_categorical((labels))

data = sequences;
MAX_SEQUENCE_LENGTH = 2000;
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH); #most sequences probably around 1000, not much more

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

VALIDATION_SPLIT = 0.2;
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

## USE GLOVE WORD EMBEDDING
print('adding a word embedding')
embeddings_index = {}
f = open(GLOVE_DIR+ 'glove.6B.100d.txt', 'r', encoding="utf8")
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
MAX_NB_WORDS = 2500; #I would bet that if we set look at the training set, too many stop words are probably there

sequence_length = MAX_SEQUENCE_LENGTH

inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False, embeddings_regularizer=regularizers.l2(1e-8))(inputs)

bid = Bidirectional(CuDNNLSTM(200))(embedding)
#td = TimeDistributed(Dense(num_labels+1, activation='softmax'))(bid)
output = Dense(num_labels+1, activation = 'softmax')(bid)
model = Model(inputs=inputs, outputs=output)

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.add(Dense(12, activation = 'relu', input_dim = (2001, 41, 1)))
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Traning Model...")
history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=True, validation_data=(x_val, y_val))  # starts training
model.save('economist_bidirectional.h5')