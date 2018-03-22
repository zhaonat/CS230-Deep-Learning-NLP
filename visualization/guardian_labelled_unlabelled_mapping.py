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

guardian_unlabelled = dir+'Guardian_raw_epoch_1_with_labels.p';
guardian = os.path.join(dir,'condensed_label_guardian_dataset.p');
guardian2 = os.path.join(dir, 'Guardian_unlabelled_corpus.p')
[guardian_labelled_corpus, guardian_labels] = pickle.load(open(guardian, 'rb'));
[guardian_unlabelled_corpus] = pickle.load(open(guardian2, 'rb'));

sample = np.random.randint(0, len(guardian_unlabelled_corpus), 10000);
guardian_unlabelled_corpus = [guardian_unlabelled_corpus[i] for i in sample];

total_corpus = guardian_labelled_corpus+guardian_unlabelled_corpus
dep_labels = [1]*len(guardian_labelled_corpus)+[0]*len(guardian_unlabelled_corpus);
# Use tf (raw term count) features to tokenize the labelled guardian corpus
n_features = 200;
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_features=n_features,
                                stop_words='english')

tf = tf_vectorizer.fit_transform(total_corpus);

tf = tf.todense();

print('fitting tsne')
X_embedded = TSNE(n_components=2, verbose = True).fit_transform(tf);


## === this is not good as it appears to show that the labelled and unlabelled texts are actually quite separable
# hence, they are not from the same distribution
plt.figure(figsize = (20,20))
plt.scatter(X_embedded[:,0], X_embedded[:,1], c= dep_labels,  edgecolor='black', s = 320, cmap = 'summer');
plt.xlabel('Embedded Dimension 1', fontsize = 40);
plt.ylabel('Embedded Dimension 2', fontsize = 40);
plt.title('t-SNE visualization of the Guardian and Deployment Articles', fontsize = 40)
class_colours = ['khaki', 'olivedrab']
import matplotlib.patches as mpatches
recs = list(); classes = ['train/test','deployment']
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
plt.legend(recs,classes,loc=4, fontsize = 30)
plt.savefig('Guardian_Deployment_t-SNE.png')
plt.show()