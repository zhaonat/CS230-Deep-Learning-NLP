import numpy as np
import collections
import gensim
import pickle
from settings import ROOT_DIR
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

''' the input for gensim word2vec is a sequence of sentences as its input.
    Each sentence a list of words (utf8 strings):'''

dir = ROOT_DIR+'\\processed_data\\';

data = pickle.load(open(dir+'sentence_text.p', 'rb'));
document = data[1];

#compiled sentences
compiled_sentences = data[0];
for i in range(len(data)):
    compiled_sentences += data[i];

print(compiled_sentences)
model = gensim.models.Word2Vec(compiled_sentences, min_count=5)

print(model.similarity('Greece', 'January'))
print(model.most_similar(positive=['woman', 'bailout'], negative=['finance'], topn=1))
X_tot = list();
for word in model.wv.vocab:
    X_tot.append(model.wv[word]);

X_tot = np.array(X_tot);
## visualize embedding using t-sne
X_embedded = TSNE(n_components=2, verbose=True, perplexity=40).fit_transform(X_tot)
X_embedded.shape

## plot the t-sne result
plt.figure()
plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
plt.show()

