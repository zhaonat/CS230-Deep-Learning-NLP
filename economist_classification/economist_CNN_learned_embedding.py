from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from keras.layers import Embedding
import settings

#find the pre-trained GLOVE
GLOVE_DIR = settings.ROOT_DIR+'\\glove.6B\\'

dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_raw.p'

[data, labels] = pickle.load(open(dir+file, 'rb'));
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


MAX_NB_WORDS = 2000;

# we should really know what Tokenizer does...
texts = data;
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH = 1500;
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
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


EMBEDDING_DIM = 300;
## learning our own embedding CLEARLY SUCKS! so using a pretrained one seems to be the optimal route...

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

## construct simple CNN arthitecture

sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

## convolutional layers

x = Conv1D(filters = 64, kernel_size = 10, activation='relu')(embedded_sequences)
x = MaxPooling1D(pool_size = 2)(x)
x = Conv1D(filters = 128, kernel_size = 5, activation='relu')(x)
x = MaxPooling1D(pool_size = 2)( x)  # global max pooling

## dense layers

x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
x = Dropout(rate = 0.2)(x);
preds = Dense(len(labels_index)+1, activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

# happy learning!
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=20, batch_size=128)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.show()