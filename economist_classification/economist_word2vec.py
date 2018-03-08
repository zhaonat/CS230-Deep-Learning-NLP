from gensim.models import word2vec
from gensim.models import KeyedVectors
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

word_vectors = KeyedVectors.load_word2vec_format('/tmp/vectors.txt', binary=False)
dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_no_process.p'
file = 'Econ_corpus_raw_10k.p'

[data, labels] = pickle.load(open(dir+file, 'rb'));
num_labels = len(set(labels))+1;

print(data)

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

model = word2vec.Word2Vec(data, size=100, window=5, min_count=5, workers=4)