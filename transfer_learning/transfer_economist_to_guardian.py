import settings
import os
import pickle
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Embedding, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Bidirectional, CuDNNLSTM
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


##=====================transfer learning infrastructure ============================== ##
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet"""
  x = base_model.output
  predictions = Dense(nb_classes, activation='softmax')(x)
  model = Model(input=base_model.input, output=predictions)
  return model

## =================================================================================== ##

model_dir = os.path.join(settings.ROOT_DIR, 'saved_models');
corpus_dir = os.path.join(settings.ROOT_DIR, 'Guardian_corpus');

#load an economist model
path = os.path.join(model_dir, 'multilayer_economist_CNN_Glove_II.h5')
path = os.path.join(model_dir, 'multilayer_economist_CNN_Glove_II.h5')

econ_model = load_model(path)
print(econ_model.summary())
#load the guardian dataset
[corpus, labels] = pickle.load(open(os.path.join(corpus_dir, 'condensed_label_guardian_dataset.p'), 'rb'));
num_labels = len(set(labels))
print(len(corpus))

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
max_features = 2000;
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
data = pad_sequences(sequences, maxlen=max_features)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# Data preparing
X_train, X_test, Y_train, Y_test = train_test_split(data,onehot_encoded,test_size=0.15)

## convert labels

#freeze the last layer
for i in range(1,4):
    econ_model.layers[i].trainable = False
econ_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']); #dumb compiles are needed every time we modify some of the loaded models?
## replace the last layer
econ_model.layers.pop();
econ_model.layers.pop() # Get rid of the dropout layer
econ_model.outputs = [econ_model.layers[-1].output]
econ_model.layers[-1].outbound_nodes = []
econ_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']);
print(econ_model.summary())

#Adding custom Layers
x = econ_model.layers[-1].output
x = Dropout(0.2, name = 'new_dropout')(x);
x = Dense(1024, activation="relu", name = 'new_dense')(x)

new_classification_layer = Dense(num_labels, activation = 'softmax', name = 'classifier_guardian')(x)

new_model = Model(inputs = econ_model.input, outputs = new_classification_layer)
print(new_model.summary())
new_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc']);

print(new_model.summary())


history = new_model.fit(X_train,Y_train, validation_split = 0.05, verbose=True, epochs = 100, batch_size = 64)