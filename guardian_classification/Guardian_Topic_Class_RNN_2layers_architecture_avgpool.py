import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras import backend
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Reshape, InputLayer, Lambda, GlobalAveragePooling1D
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import regularizers
from keras import optimizers
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import argparse, os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', type=str, help='directory containing data', default='../NewsMining/no_stop_words/')
parser.add_argument('-o', '--output', type=str, help='path to save trained model', default='models/model.h5')
parser.add_argument('-p', '--pretrain', type=str, help='path to pretrained model', default=None)
parser.add_argument('-e', '--epochs', type=int, help='training epochs', default=300)
parser.add_argument('-w', '--max_words', type=int, help='max_words in a sentence', default=20)
parser.add_argument('-s', '--max_sentences', type=int, help='max_sentences in a articles', default=20)
parser.add_argument('-v', '--word_vector_dim', type=int, default=128)
parser.add_argument('--LSTM1_dims', type=int, nargs='+', default=[128, 128])
parser.add_argument('--LSTM2_dims', type=int, nargs='+', default=[128, 128])
parser.add_argument('--DENSE_dims', type=int, nargs='*', default=[128])
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--d', type=float, default=0., help='dropout rate')
parser.add_argument('--rd', type=float, default=0., help='recurrent dropout rate')
parser.add_argument('--dd', type=float, default=0., help='dense layer dropout rate')
args = parser.parse_args()
print(args)

max_words = args.max_words  # number of words in a sentence
max_sentences = args.max_sentences # number of sentences in an article
word_vector_dim = args.word_vector_dim
LSTM1_dims = args.LSTM1_dims
LSTM2_dims = args.LSTM2_dims
DENSE_dims = args.DENSE_dims
lr = args.lr
dropout = args.d
recurrent_dropout = args.rd

data = np.load(os.path.join(args.input_dir, 'docs.npy'))
onehot_encoded = np.load(os.path.join(args.input_dir, 'labels.npy'))

### model
num_word = 1000
num_labels = len(np.unique(onehot_encoded, axis=0))
print(num_labels)
batch_size = 10

def backend_reshape(x, shape):
    return keras.backend.reshape(x, shape)

model = Sequential()
model.add(InputLayer(input_shape=(max_sentences, max_words)))
model.add(Lambda(backend_reshape, output_shape=(max_words,), arguments={'shape': (-1, max_words)}))

model.add(Embedding(num_word, word_vector_dim, input_length=max_words))

## LSTM_1 for sentences

#model.add(Dense(12, activation = 'relu', input_dim = (2001, 41, 1)))
for d in LSTM1_dims:
    model.add(LSTM(d, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
model.add(GlobalAveragePooling1D()) ## Global Pooling

## LSTM_2 for articles
#model.add(Reshape((batch_size, max_sentences, max_words), input_shape=(batch_size*max_sentences, max_words)))
model.add(Lambda(backend_reshape, output_shape=(max_sentences, LSTM1_dims[-1]), arguments={'shape': (-1, max_sentences, LSTM1_dims[-1])}))
for d in LSTM2_dims:
    model.add(LSTM(d, dropout=dropout, recurrent_dropout=recurrent_dropout, return_sequences=True))
model.add(GlobalAveragePooling1D()) ## Global Pooling

for d in DENSE_dims:
    model.add(Dropout(args.dd))
    model.add(Dense(d, activation='relu'))
model.add(Dense(num_labels, activation='softmax'))

opt = optimizers.adam(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


#batch size constrained by memory of the gpu...which is sad

print(data.shape, onehot_encoded.shape)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

if args.pretrain is not None:
    model.load_weights(args.pretrain)
    print("loaded pretrained model: %s" % args.pretrain)
history = model.fit(data, onehot_encoded, epochs=args.epochs, batch_size=batch_size,  validation_split=0.2, verbose = True);
print("model saved to %s" % args.output)
model.save_weights(args.output)



#save model data
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Guardian topic RNN 2-Layer-architecture-model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Guardian_topic_RNN_model_accuracy.pdf');
# summarize history for loss


plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Guardian topic RNN 2-Layer-architecture-model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
filename = 'logs/Guardian_topic_RNN_model_loss_lr_%f_w_%d_l1d_%s_l2d_%s_dd_%s.pdf' % \
	    (lr, word_vector_dim, '-'.join(str(x) for x in LSTM1_dims),
	    '-'.join(str(x) for x in LSTM2_dims),
	    '-'.join(str(x) for x in DENSE_dims))
plt.savefig(filename)

print(args)
print("save to %s" % filename)
