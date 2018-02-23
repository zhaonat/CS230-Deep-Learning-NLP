import os
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from settings import ROOT_DIR
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

dir = os.getcwd()
data_dir = ROOT_DIR+'\\financial_times\\FT-archive-2013\\';
out_dir = ROOT_DIR+'\\processed_data\\';
print(dir)
counter = 0;
document_data = list(); #stores all texts as single string
word_sentence_data = list(); #stores text as individual words found in the article


## first read all the documents into the corpus as raw data (no editing)
subject_tags = list();
for file in os.listdir(data_dir):

    data = json.load(open(data_dir+file,  encoding="utf8"));
    #pprint(data)

    text = data['bodyXML']
    annotations_tags = data['annotations'];
    #cycle through annotations
    tags = list();
    for annote in annotations_tags:
        if(annote['type'] == 'SUBJECT'):
            tags.append(annote['prefLabel']);
    subject_tags.append(tags);

    #tags = data[];
    if(counter%1000 == 0):
        print('file: '+str(counter));
        print(data['types'])

    #clean html from the document while it is still a string
    text = remove_tags(text);
    document_data.append(text);

    counter+=1;
    if(counter> 1000):
        break;

## tokenization step
tokenized_docs = [word_tokenize(doc) for doc in document_data]
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
tokenized_docs_no_stopwords = []
for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    tokenized_docs_no_stopwords.append(new_term_vector)

print(tokenized_docs_no_stopwords);

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
file = open(out_dir+'FT_corpus_2013.p', 'wb')
pickle.dump(preprocessed_docs, file);

file = open(out_dir+'FT_raw_corpus_2013.p', 'wb')
pickle.dump(raw_preprocessed, file);