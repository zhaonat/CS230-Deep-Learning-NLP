from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, merge
from keras.layers import Reshape, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
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
from keras.layers import Embedding, Layer
from keras import backend as K
from keras import initializers

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        # self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai / K.sum(ai, axis=1).dimshuffle(0, 'x')

        weighted_input = x * weights.dimshuffle(0, 1, 'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

#find the pre-trained GLOVE
GLOVE_DIR = settings.ROOT_DIR+'\\word_embeddings\\glove.6B\\' #when feeding word embeddings, don't lemmatize
econ_dir = settings.ROOT_DIR+'\\economist_corpus\\'
dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_no_process.p'
#file = 'Econ_corpus_raw_10k.p'

[data, labels] = pickle.load(open(dir+file, 'rb'));

### FILTER OUT SMALL LENGTH DATA SAMPLES
remove_index = list();
num_removed = 0;
for i in range(len(labels)):
    if(len(data[i]) < 1000):
        remove_index.append(i);
        num_removed+=1;
data = [i for j, i in enumerate(data) if j not in remove_index];
labels = [i for j, i in enumerate(labels) if j not in remove_index];
print('number of articles removed: '+str(num_removed))
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
num_labels = len(set(labels));


MAX_NB_WORDS = 2500; #I would bet that if we set look at the training set, too many stop words are probably there

MAX_SEQUENCE_LENGTH = 2000; #largest sequence is 25000...
sequence_length = MAX_SEQUENCE_LENGTH

# we should really know what Tokenizer does...
texts = data;
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(texts)
# sequences = tokenizer.texts_to_sequences(texts)
#
# ## save the tokenizer for future use
# handle = open('economist_tokens_100k.p', 'wb');
# pickle.dump([tokenizer, sequences], handle, protocol=pickle.HIGHEST_PROTOCOL)
[tokenizer, sequences] = pickle.load(open(econ_dir+'economist_tokens_100k.p', 'rb'))

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

VALIDATION_SPLIT = 0.05;
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

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

print('Found %s word vectors.' % len(embeddings_index))

## CNN params
#embedding_dim = 64 # can be changed
filter_sizes = [2,3,4,5,6];
num_filters = [200,200,200,400, 400, 100]
drop = 0.3
epochs = 25
batch_size = 128
print("Create CNN Model")

## using word glove, but make it TRAINABLE
inputs = Input(shape=(sequence_length,), dtype='int32')
embedding = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True, embeddings_regularizer=regularizers.l2(1e-8))(inputs)

## regularize the word embedding to make dimensionality more favorable
conv_0 = Conv1D(num_filters[0], kernel_size=(filter_sizes[0]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_0 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), strides=1, padding='valid')(conv_0)
pooled_conv_0 = Dropout(drop)(maxpool_0)
conv_1 = Conv1D(num_filters[1], kernel_size=(filter_sizes[1]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_1 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[1] + 1), strides=1, padding='valid')(conv_1)
pooled_conv_1 = Dropout(drop)(maxpool_1)
conv_2 = Conv1D(num_filters[2], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_2 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_2)
pooled_conv_2 = Dropout(drop)(maxpool_2)
conv_3 = Conv1D(num_filters[3], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_3 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_3)
pooled_conv_3 = Dropout(drop)(maxpool_2)
conv_4 = Conv1D(num_filters[4], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal', activation='relu')(embedding)
maxpool_4 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_4)
pooled_conv_4 = Dropout(drop)(maxpool_2)

#concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
concatenated_tensor = Concatenate(axis=1)([pooled_conv_0, pooled_conv_1, pooled_conv_2 ])
print(concatenated_tensor.shape)
flatten = Flatten()(concatenated_tensor)
## pass this through the attention module
#l_dense = TimeDistributed(Dense(200))(flatten)
l_att = Activation(activation='softmax')(flatten);

attention_add =  merge([flatten, l_att], output_shape=32, name='attention_mul', mode='mul')


## ============== DENSE STRUCTURE OF NEURAL NET ====================
#dense1 = Dense(128, activation = 'relu')(dropout)
## ==================================================================

output = Dense(num_labels+1, activation='softmax')(attention_add)
model = Model(inputs=inputs, outputs=output)

adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.add(Dense(12, activation = 'relu', input_dim = (2001, 41, 1)))
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

print("Training Model...")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(x_val, y_val))  # starts training
model.save('attention_CNN.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.show()

