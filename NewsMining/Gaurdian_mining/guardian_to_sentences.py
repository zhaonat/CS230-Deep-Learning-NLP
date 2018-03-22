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

regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

dir =  settings.ROOT_DIR+'\\Guardian_corpus\\';
economist = dir+'\\Guardian_10K_epoch_1.csv';

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
    type_label.append(corT[article]['sectionId'])
    counter+=1;
    if(counter%1000 == 0):
        print(counter)
    # if(counter> 10000):
    #     break;


## save the corpus (just a list of lists of words in the document
file = open(dir+'Guardian_corpus_raw.p', 'wb')
pickle.dump([corpus,  type_label], file);


sentence_corpus = list();
for doc in corpus:
    sentences = doc.split('.')
    print(len(sentences))
    sentence_corpus.append(sentences);

## save the corpus (just a list of lists of words in the document
file = open(dir+'Guardian_corpus_sentences.p', 'wb')
pickle.dump([sentence_corpus, type_label], file);




