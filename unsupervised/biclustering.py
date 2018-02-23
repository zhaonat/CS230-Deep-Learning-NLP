from __future__ import print_function

from collections import defaultdict
import operator
from time import time

import numpy as np
import pickle
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.externals.six import iteritems
from sklearn.feature_extraction.text import TfidfVectorizer
from settings import ROOT_DIR
print(__doc__)


def number_normalizer(tokens):
    """ Map all numeric tokens to a placeholder.

    For many applications, tokens that begin with a number are not directly
    useful, but the fact that such a token exists can be relevant.  By applying
    this form of dimensionality reduction, some methods may perform better.
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))


dir = ROOT_DIR+'\\processed_data\\'
data = pickle.load(open(dir+'FT_raw_corpus_2013.p', 'rb'));

vectorizer = NumberNormalizingVectorizer(stop_words='english', min_df=5)
cocluster = SpectralCoclustering(n_clusters= 20,
                                 svd_method='arpack', random_state=0)
kmeans = MiniBatchKMeans(n_clusters=20, batch_size=20000,
                         random_state=0)

print("Vectorizing...")
X = vectorizer.fit_transform(data)

print("Coclustering...")
start_time = time()
cocluster.fit(X)
y_cocluster = cocluster.row_labels_

print("MiniBatchKMeans...")
start_time = time()
y_kmeans = kmeans.fit_predict(X)

feature_names = vectorizer.get_feature_names()



