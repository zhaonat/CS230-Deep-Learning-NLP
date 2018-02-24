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

dir = settings.ROOT_DIR+'\\processed_data\\'

guardian = dir+'Guardian_raw_epoch_1_with_labels.p';
guardian = dir+'condensed_label_guardian_dataset.p';

[guardian_corpus, labels] = pickle.load(open(guardian, 'rb'));
print(len(set(labels)))
#process every article

#convert labels to one-hot encoding
le = preprocessing.LabelEncoder()
le.fit(labels);
y = le.fit_transform(labels);



# Use tf (raw term count) features for LDA.
n_features = 200;
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(guardian_corpus);

tf = tf.todense();

print('fitting tsne')
X_embedded = TSNE(n_components=2, verbose = True).fit_transform(tf);
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y, cmap='jet_r');
plt.xlabel('Embedded Dimension 1');
plt.ylabel('Embedded Dimension 2');
plt.title('t-SNE visualization of the tf-idf scores for the Guardian')
plt.show()