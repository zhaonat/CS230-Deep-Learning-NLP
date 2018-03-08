#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:15:26 2018

@author: Jiahui
"""

#import settings
#ROOT_DIR = os.path.dirname(os.path.abspath('/Users/Jiahui/Classes/2018Winter/cs230/project/code/CS230'))
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPool2D
from keras.layers import Reshape, Flatten, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Model
from collections import Counter

max_features = 4000;

dir = 'D:/Documents/Classes/CS230/processed_data/'
GLOVE_DIR = 'D:/Documents/Classes/CS230/embeddings/glove.6B/'
guardian = dir+'Guardian_raw_epoch_1_with_labels_no_lemma.p';
#guardian = dir+'Guardian_epoch_1_with_labels.p';

# process data to make it in the
[guardian_corpus, labels] = pickle.load(open(guardian, 'rb'));
count_label = Counter(labels)
for i in range(len(guardian_corpus)):
    if Counter(labels)[labels[i]] < 50:
        labels[i] = 'other'

#convert labels to one hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
print('size of data: '+str(len(integer_encoded)));
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# plt.hist(integer_encoded);
# plt.show()

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
num_labels = len(set(labels));

# Use tokenizer
num_word = 3000 # will be the input dimensions
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(guardian_corpus)
sequences = tokenizer.texts_to_sequences(guardian_corpus)
data = pad_sequences(sequences, maxlen=max_features)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# Data preparing
X_train, X_test, Y_train, Y_test = train_test_split(data,onehot_encoded,test_size=0.15)
sequence_length = X_train.shape[1]


## construct word embedding before feeding it into CNN (pre-trained)
print('construct embedding from GloVE: this takes a while')
embeddings_index = {}
f = open(GLOVE_DIR+ 'glove.6B.200d.txt', 'r', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
EMBEDDING_DIM = 200; #this is FIXED by 100d in the GLoVE model

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

## probably saving this is a smart idea


filter_sizes = [3,4,5]
num_filters = [100,100,100]
drop = 0.3

epochs = 100
batch_size = 100

## this is our features
# now fit a neural net
print("Create CNN Model")
MAX_SEQUENCE_LENGTH = max_features;
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

## construct simple CNN arthitecture

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

## some significant hyperparameter optimization will likely help
x = Conv1D(64, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 10, activation='relu')(x)
x = MaxPooling1D(2)(x)  # global max pooling
x = Flatten()(x)
x = Dense(500, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(rate = 0.2)(x);
x = Dense(200, activation = 'relu', kernel_regularizer = 'l2')(x)
x = Dropout(rate = 0.2)(x);

preds = Dense((num_labels), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# happy learning!
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
          epochs=20, batch_size=128)

#save model data
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(' Guardian topic CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(' Guardian topic CNN model accuracy.png');# summarize history for loss



plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Guardian topic CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show();


