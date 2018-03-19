from __future__ import print_function

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

dir = settings.ROOT_DIR+'\\processed_data\\'
econ_dir = os.path.join(settings.ROOT_DIR, 'economist_corpus');
guardian = dir+'Guardian_raw_epoch_1_with_labels.p';
guardian = dir+'condensed_label_guardian_dataset.p';
econ = os.path.join(econ_dir,'econ_corpus_raw.p');

#[guardian_corpus, labels] = pickle.load(open(guardian, 'rb'));
[guardian_corpus, labels] = pickle.load(open(econ, 'rb'));

print(guardian_corpus[0])
print(len(set(labels)))
#process every article

#convert labels to one-hot encoding
le = preprocessing.LabelEncoder()
le.fit(labels);
y = le.fit_transform(labels);

# Use tf (raw term count) features for LDA.
n_features = 1000;
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_features=n_features,
                                stop_words='english')
t0 = time()
tf = tf_vectorizer.fit_transform(guardian_corpus);

print("done in %0.3fs." % (time() - t0))
print()
X_train,X_test, y_train,y_test = train_test_split(tf.todense(), y, test_size = 0.2);
print(X_train.shape)

clf = MultinomialNB()
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

cm = confusion_matrix(clf.predict(X_test), y_test);

#PCA of td-idf
pca = PCA(n_components = 2);
pca.fit(X_train);
X_2 = pca.fit_transform(scale(X_train));

plt.scatter(X_2[:,0], X_2[:,1], c = y_train);
plt.show()

