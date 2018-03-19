import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import string
import pandas as pd
import settings
'''
if an article is too long, we will split it into two articles with the same label.
'''
regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

dir =  settings.ROOT_DIR+'\\economist_corpus\\';
economist = dir+'\\Economist.csv';

corpora = pd.read_csv(economist, index_col = 0, encoding = "ISO-8859-1");
print(corpora.columns)
#print(corpora)

corT = corpora.T;
corpus = list();

counter = 0;
type_label = list();
for article in corT.columns:
    text = corT[article]['text'];
    #print(text)
    if(str(text) == 'nan'):
        continue;
    corpus.append(text)
    type_label.append(corT[article]['section_name'])
    counter+=1;
    if(counter%1000 == 0):
        print(counter)
    # if(counter> 10000):
    #     break;

split_corpus = list(); split_corpus_labels = list();
splits = 2; length_cutoff = 400
word_length_dist = list();
for i in range(len(corpus)):
    document = corpus[i];
    words = (document.split(' '));
    word_length_dist.append(len(words))
    if(len(words) > length_cutoff):
        split_size = int(length_cutoff/splits)
        for j in range(splits):
            subset = words[j*split_size:(j+1)*split_size];
            sub_doc = ' '. join(subset)
            print(sub_doc)
            split_corpus.append(sub_doc);
            split_corpus_labels.append(type_label[i])
    else:
        split_corpus.append(document);
        split_corpus_labels.append(type_label[i])

pickle.dump([split_corpus, split_corpus_labels], open('Econ_reduced_length_corpus.p', 'wb'));