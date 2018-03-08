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

regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

dir =  'D:\\StanfordYearTwo\\Classes\\CS230_Learning\\data_files\\';
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
    if(counter> 10000):
        break;

## tokenization step
tokenized_docs = [word_tokenize(doc) for doc in corpus]
print(tokenized_docs)

## remove punctuation
tokenized_docs_no_punctuation = []

for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)

    tokenized_docs_no_punctuation.append(new_review)
print(tokenized_docs_no_punctuation);

## remove stop words
# tokenized_docs_no_stopwords = []
# for doc in tokenized_docs_no_punctuation:
#     new_term_vector = []
#     for word in doc:
#         if not word in stopwords.words('english'):
#             new_term_vector.append(word)
#     tokenized_docs_no_stopwords.append(new_term_vector)
#
# print(tokenized_docs_no_stopwords)
tokenized_docs_no_stopwords  = tokenized_docs_no_punctuation;

## stemming and lemmatizing
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = []

for doc in tokenized_docs_no_stopwords:
    final_doc = []
    for word in doc:
        final_doc.append(porter.stem(word))
        #final_doc.append(snowball.stem(word))
        #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
    preprocessed_docs.append(final_doc)

print(preprocessed_docs)
## convert preprocessed doc back into sentences

raw_preprocessed = [' '.join(x) for x in preprocessed_docs];

## save the corpus (just a list of lists of words in the document
file = open(dir+'Econ_corpus_preprocessed.p', 'wb')
pickle.dump([preprocessed_docs, type_label], file);

file = open(dir+'Econ_corpus_raw_10k.p', 'wb')
pickle.dump([raw_preprocessed, type_label], file);

