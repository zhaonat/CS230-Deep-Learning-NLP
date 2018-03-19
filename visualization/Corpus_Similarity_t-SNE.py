from sklearn.manifold import TSNE

import settings
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.naive_bayes import MultinomialNB
from time import time
from sklearn.model_selection import train_test_split
from settings import ROOT_DIR
from sklearn.metrics import confusion_matrix;
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.preprocessing import scale

import pickle
from sklearn.decomposition import PCA, NMF
import os
import numpy as np
import pandas as pd

dir = settings.ROOT_DIR+'\\Guardian_corpus\\'
econ_dir = os.path.join(settings.ROOT_DIR, 'economist_corpus')

guardian = dir+'Guardian_raw_epoch_1_with_labels.p';
guardian = os.path.join(dir,'condensed_label_guardian_dataset.p');
econ = os.path.join(econ_dir,'Econ_corpus_raw.p')

[guardian_corpus, guardian_labels] = pickle.load(open(guardian, 'rb'));
[econ_corpus, econ_labels] = pickle.load(open(econ, 'rb'));


#sample 10000 documents from econ_corpus so we don't wash out the guardian
sample = np.random.randint(0, len(econ_labels), 10000)
econ_corpus = [econ_corpus[index] for index in sample];

corpus_label = [1]*10000+[0]*len(guardian_labels)
total_corpus = econ_corpus+guardian_corpus;

# Use tf (raw term count) features for LDA.
n_features = 2000;
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(total_corpus);

tf = tf.todense();

print('fitting tsne')
X_embedded = TSNE(n_components=2, verbose = True).fit_transform(tf);

data = pd.DataFrame(X_embedded);
data['labels'] = corpus_label
data.to_csv('guardian_vs_econ_t-SNE.csv');

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=corpus_label, cmap='jet_r');
plt.xlabel('Embedded Dimension 1');
plt.ylabel('Embedded Dimension 2');
plt.title('t-SNE visualization of the Economist vs the Guardian')
plt.show()