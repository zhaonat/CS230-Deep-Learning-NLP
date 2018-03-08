from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import Reshape, Flatten, Concatenate
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from keras import regularizers
from keras.optimizers import Adam
import settings
from keras.layers import Embedding

dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_raw_10k.p'

[data, labels] = pickle.load(open(dir+file, 'rb'));
print(data)

#get the max sequence length from data:
MAX_SEQUENCE_LENGTH = 0;
for i in data:
    if(len(i) > MAX_SEQUENCE_LENGTH):
        MAX_SEQUENCE_LENGTH = len(i);
num_labels = len(set(labels))+1; #there's a mistake here

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


MAX_NB_WORDS = 2500; #I would bet that if we set look at the training set, too many stop words are probably there
num_word = MAX_NB_WORDS # will be the input dimensions

#MAX_SEQUENCE_LENGTH = 1500;
sequence_length = MAX_SEQUENCE_LENGTH
# we should really know what Tokenizer does...
texts = data;
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH); #most sequences probably around 1000, not much more

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
print(data)
labels = labels[indices]