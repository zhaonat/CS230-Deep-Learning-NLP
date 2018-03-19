import settings
from keras.models import load_model
import os
from activation_visualization.activation_extractors import *
import pickle
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

#load a model
model_name = 'guardian_CNN.h5';
model_dir = os.path.join(settings.ROOT_DIR, 'saved_models')
econ_dir= os.path.join(settings.ROOT_DIR, 'economist_corpus')
path = os.path.join(model_dir, model_name)
model = load_model(path);

print(model.summary())

#load a corpus
[corpus, labels] = pickle.load(open(os.path.join(econ_dir, 'Econ_corpus_raw.p'), 'rb'))

# Use tokenizer
num_word = 3000 # will be the input dimensions
max_features = 4000;
tokenizer = Tokenizer(num_words=num_word)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
data = pad_sequences(sequences, maxlen=max_features)

import numpy as np
import matplotlib.pyplot as plt
input = data[0,:];
input = np.reshape(input, (1,4000))
## get activations
activations = get_activations(model, input, print_shape_only=False, layer_name=None)
print(activations)

#get activation geometries
for act_layer in activations:
    print(act_layer.shape)

activation_sample = activations[1][0,:]
plt.imshow(activation_sample)
plt.show()
display_activations(activation_sample)
