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

guardian = dir+'Guardian_raw_epoch_1_with_labels.p';
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

# Data preparing
X_train, X_test, Y_train, Y_test = train_test_split(data,onehot_encoded,test_size=0.15)
sequence_length = X_train.shape[1]
embedding_dim = 64 # can be changed
filter_sizes = [3,4,5]
num_filters = [100,100,100]
drop = 0.3

epochs = 25
batch_size = 100

## this is our features
# now fit a neural net
print("Create CNN Model")
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(input_dim=num_word, output_dim=embedding_dim, input_length=sequence_length)(inputs)

conv_0 = Conv1D(num_filters[0], kernel_size=(filter_sizes[0]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_0 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), strides=1, padding='valid')(conv_0)
pooled_conv_0 = Dropout(drop)(maxpool_0)
conv_1 = Conv1D(num_filters[1], kernel_size=(filter_sizes[1]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_1 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[1] + 1), strides=1, padding='valid')(conv_1)
pooled_conv_1 = Dropout(drop)(maxpool_1)
conv_2 = Conv1D(num_filters[2], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_2 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_2)
pooled_conv_2 = Dropout(drop)(maxpool_2)

#concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
concatenated_tensor = Concatenate(axis=1)([pooled_conv_0, pooled_conv_1, pooled_conv_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(drop)(flatten)
output = Dense(num_labels,kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001), activation='softmax')(dropout)
model = Model(inputs=inputs, outputs=output)

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.add(Dense(12, activation = 'relu', input_dim = (2001, 41, 1)))
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Traning Model...")
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(X_test, Y_test))  # starts training

plt.figure(figsize = (20,15))


plt.rc('axes', linewidth=3)
plt.rc('axes', )
plt.rc('font', size=30)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=35)    # fontsize of the tick labels
plt.rc('ytick', labelsize=35)    # fontsize of the tick labels
plt.rc('legend', fontsize=30)    # legend fontsize
plt.rc('figure', titlesize=55)  # fontsize of the figure title


#save model data
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'], linewidth = 2)
plt.plot(history.history['val_acc'], linewidth = 2)
plt.title(' Guardian topic CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Guardian topic CNN model accuracy.png');# summarize history for loss



plt.figure(figsize = (20,15))
plt.plot(history.history['loss'], linewidth = 2)
plt.plot(history.history['val_loss'], linewidth = 2)
plt.title('Guardian topic CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Guardian topic CNN Loss History.png');# summarize history for loss

plt.show();

model.summary();

