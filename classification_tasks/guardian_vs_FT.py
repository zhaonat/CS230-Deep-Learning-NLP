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

max_features = 3000;

dir = settings.ROOT_DIR+'\\processed_data\\'

guardian = dir+'Guardian_raw_epoch_1.p';
financial = dir+'FT_raw_corpus_2013.p';

guardian_corpus = pickle.load(open(guardian, 'rb'));
financial_corpus = pickle.load(open(financial, 'rb'));

##generate labels
labels = np.array([0]*len(guardian_corpus)+[1]*len(financial_corpus));

##merge the two corpuses
corpus = guardian_corpus+financial_corpus;

# Use tokenizer
num_word = 2000
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
data = pad_sequences(sequences, maxlen=max_features)

## this is our features
# now fit a neural net

model = Sequential()
model.add(Embedding(num_word, 32, input_length=max_features)) #embedding makes memory constraings difficult
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#batch size constrained by memory of the gpu...which is sad
history = model.fit(data, labels, epochs=5, batch_size=100,  validation_split=0.3, verbose = True);

#save model data
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('FT vs Guardian RNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('FT vs Guardian RNN model accuracy.png');
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('FT vs Guardian RNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show();

