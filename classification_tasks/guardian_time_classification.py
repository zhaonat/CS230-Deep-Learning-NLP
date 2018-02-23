import settings
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers

## essentially the net is not able to discern anything in regards to time

max_features = 300;

dir = settings.ROOT_DIR+'\\processed_data\\'

guardian = dir+'Guardian_raw_epoch_4_with_month_labels.p';

[guardian_corpus, labels] = pickle.load(open(guardian, 'rb'));

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
num_word = 3000
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(guardian_corpus)
sequences = tokenizer.texts_to_sequences(guardian_corpus)
data = pad_sequences(sequences, maxlen=max_features)

## this is our features
# now fit a neural net

model = Sequential()
model.add(Embedding(num_word, 64, input_length=max_features)) #embedding makes memory constraings difficult
#num_word is for:
#max_features is for:

model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout =0.2))
model.add(Dense(40, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#batch size constrained by memory of the gpu...which is sad
history = model.fit(data, onehot_encoded, epochs=100, batch_size=100,  validation_split=0.2, verbose = True, shuffle = True);

#save model data
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title(' Guardian topic RNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(' Guardian topic RNN model accuracy.png');
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Guardian topic RNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show();

