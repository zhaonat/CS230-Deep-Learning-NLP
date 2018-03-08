
## hyperparameter optimization on regularization strengths
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
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers
from keras.optimizers import Adam
import settings
from keras.layers import Embedding

dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_raw_10k.p'

[data, labels] = pickle.load(open(dir+file, 'rb'));
print(data)
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

MAX_SEQUENCE_LENGTH = 4000;
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

## =======================================================================================================##
def create_model(neurons=1, embedding_regularizer = 1e-3):
    # create model
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding = Embedding(input_dim=num_word, embeddings_regularizer=regularizers.l2(embedding_regularizer),
                          output_dim=embedding_dim, input_length=sequence_length)(inputs)

    conv_0 = Conv1D(num_filters[0], kernel_size=(filter_sizes[0]), padding='same', kernel_initializer='normal',
                    activation='relu')(embedding)
    maxpool_0 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[0] + 1), strides=1, padding='valid')(conv_0)
    pooled_conv_0 = Dropout(drop)(maxpool_0)
    conv_1 = Conv1D(num_filters[1], kernel_size=(filter_sizes[1]), padding='same', kernel_initializer='normal',
                    activation='relu')(embedding)
    maxpool_1 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[1] + 1), strides=1, padding='valid')(conv_1)
    pooled_conv_1 = Dropout(drop)(maxpool_1)
    conv_2 = Conv1D(num_filters[2], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal',
                    activation='relu')(embedding)
    maxpool_2 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_2)
    pooled_conv_2 = Dropout(drop)(maxpool_2)
    conv_3 = Conv1D(num_filters[3], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal',
                    activation='relu')(embedding)
    maxpool_3 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_3)
    pooled_conv_3 = Dropout(drop)(maxpool_3)
    conv_4 = Conv1D(num_filters[4], kernel_size=(filter_sizes[2]), padding='same', kernel_initializer='normal',
                    activation='relu')(embedding)
    maxpool_4 = MaxPooling1D(pool_size=(sequence_length - filter_sizes[2] + 1), strides=1, padding='valid')(conv_4)
    pooled_conv_4 = Dropout(drop)(maxpool_4)

    # concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    concatenated_tensor = Concatenate(axis=1)(
        [pooled_conv_0, pooled_conv_1, pooled_conv_2, pooled_conv_3, pooled_conv_4])
    flatten = Flatten()(concatenated_tensor)
    dropout = Dropout(drop)(flatten)
    dense1 = Dense(neurons, activation='relu')(dropout)
    output = Dense(num_labels, kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001),
                   activation='softmax')(dense1)
    model = Model(inputs=inputs, outputs=output)
    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    # model.add(Dense(12, activation = 'relu', input_dim = (2001, 41, 1)))
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


VALIDATION_SPLIT = 0.1;
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

## CNN params
embedding_dim = 64 # can be changed
filter_sizes = [2,3,4,5,6];
num_filters = [100,100,100,100, 100, 100]
drop = 0.3

epochs = 10
batch_size = 100
print("Create CNN Model")

model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=100, verbose=1)
# define the grid search parameters
# it appears that the range of hyperparameters isn't really large enough to justify using it
neurons = [10, 15, 20, 25, 30]
embedding_reg = [1e-7, 1e-5, 1e-3];
param_grid = dict(neurons=neurons, embedding_regularizer = embedding_reg);
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))