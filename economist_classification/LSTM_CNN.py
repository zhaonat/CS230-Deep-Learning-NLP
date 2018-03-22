
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.layers import Reshape, Flatten, Concatenate
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten, Dropout, CuDNNLSTM, LSTM, CuDNNGRU, Bidirectional
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

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess = tf.Session(config = config)


dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_no_process.p'
#file = 'Econ_corpus_raw_10k.p'

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

MAX_NB_WORDS = 2500; #I would bet that if we set look at the training set, too many stop words are probably there
num_word = MAX_NB_WORDS # will be the input dimensions

MAX_SEQUENCE_LENGTH = 1500;
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
labels = labels[indices]

VALIDATION_SPLIT = 0.1;
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

## CNN params
embedding_dim = 64 # can be changed
drop = 0.3

epochs = 25
batch_size = 100
print("Create fast LSTM Model")
model = Sequential()

model.add(Embedding(input_dim=num_word, output_dim=embedding_dim, input_length=sequence_length))
model.add(Bidirectional(CuDNNLSTM(128,return_sequences=True )))
model.add(Conv1D(filters = 2, kernel_size = 2))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))
adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(x_train, y_train, batch_size=128, epochs=32, validation_split = 0.1)
score = model.evaluate(x_val, y_val)
print('test score: '+str(score))

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.figure()
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_lacc'])
# plt.show()

model.save('fast_LSTM_weights.h5')