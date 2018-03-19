import settings
import os
import pickle
from keras.models import load_model

model_dir = os.path.join(settings.ROOT_DIR, 'saved_models');
corpus_dir = os.path.join(settings.ROOT_DIR, 'Guardian_corpus');

#load an economist model
path = os.path.join(model_dir, 'economist_bidirectional.h5')
econ_model = load_model(path)

#load the guardian dataset
[corpus, labels] = pickle.load(open(os.path.join(corpus_dir, 'condensed_label_guardian_dataset.p'), 'rb'));
print(len(corpus))