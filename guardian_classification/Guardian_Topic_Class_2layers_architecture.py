import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import backend
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Reshape, InputLayer, Lambda
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers

max_words = 20  # number of words in a sentence
max_sentences = 20 # number of sentences in an article

guardian = 'Guardian_epoch_1_with_labels.p'

[guardian_corpus, labels] = pickle.load(open(guardian, 'rb'));


### pad/truncate sentences to make sure each article has same number of sentences
guardian_pad_sent = []
for article in guardian_corpus:
    if len(article) == max_sentences:
        guardian_pad_sent.append(article)
    elif len(article) > max_sentences:
        guardian_pad_sent.append(article[0:max_sentences])
    else:
        new_article = [' '] * (max_sentences - len(article)) + article
        guardian_pad_sent.append(new_article)

guardian_corpus = guardian_pad_sent


### merge labels with less than 50 samples to be "others"
labels = [x.encode('UTF8') for x in labels]
labels_dict = dict((x,labels.count(x)) for x in set(labels))
others = []
for label in labels_dict:
    if labels_dict[label] < 50:
        others.append(label)

new_labels = ['others' if x in others else x for x in labels]
labels = labels
num_labels = len(set(labels))


### convert labels to one hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


### tokenize and pad/truncate words to make sure each sentence (not article) has same number of words
num_word = 1000
tokenizer = Tokenizer(num_words=num_word)

data = []

for article in guardian_corpus:
    #print(article)
    tokenizer.fit_on_texts(article)
    sequences = tokenizer.texts_to_sequences(article)

    # pad/truncate words to make sure each sentence has same number of words
    new_article = pad_sequences(sequences, maxlen=max_words, padding='pre', truncating='pre')
    data.append(new_article)
    #print(type(new_article), new_article.shape)
    #print(new_article)


### model
batch_size = 10

model = Sequential()
model.add(InputLayer(input_shape=(max_sentences, max_words)))
def backend_reshape(x, shape):
    return backend.reshape(x, shape)
model.add(Lambda(backend_reshape, output_shape=(max_words,), arguments={'shape': (-1, max_words)}))

model.add(Embedding(num_word, 64, input_length=max_words))

## LSTM_1 for sentences

#model.add(Dense(12, activation = 'relu', input_dim = (2001, 41, 1)))
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1)) ## stacking LSTM layers

## LSTM_2 for articles
#model.add(Reshape((batch_size, max_sentences, max_words), input_shape=(batch_size*max_sentences, max_words)))
model.add(Lambda(backend_reshape, output_shape=(max_sentences, 32), arguments={'shape': (-1, max_sentences, 32)}))
model.add(LSTM(64, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))
model.add(LSTM(32, dropout=0.1, recurrent_dropout=0.1))

model.add(Dense(num_labels, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#batch size constrained by memory of the gpu...which is sad
data = np.stack(data, axis = 0)
history = model.fit(data, onehot_encoded, epochs=10, batch_size=batch_size,  validation_split=0.2, verbose = True);



#save model data
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Guardian topic RNN 2-Layer-architecture-model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(' Guardian topic RNN model accuracy.png');
# summarize history for loss


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Guardian topic RNN 2-Layer-architecture-model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show();

