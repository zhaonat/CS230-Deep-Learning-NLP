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


dir = ROOT_DIR+'\\financial_times\\'
out_dir = ROOT_DIR+'\\processed_data\\'
year_sample = 2500;

years = [2008, 2009, 2010, 2013];
document_data = list();
year_label = list();
doc_counter = 0; no_text_counter = 1;
for year in years:
    print(str(year) + ', docs: '+str(doc_counter));
    year_dir = dir+'FT-archive-'+str(year)+'\\';
    sample_counter = 0;
    for file in os.listdir(year_dir):
        data = json.load(open(year_dir + file, encoding="utf8"));
        text = data['bodyXML'];

        if (isinstance(text, str)):
            document_data.append(text);
            time = data['publishedDate'];
            year = time.split('-')[0]
            year_label.append(year);
            doc_counter += 1
        else:
            print(str(text) + ', ' + str(no_text_counter));
            no_text_counter += 1;

        sample_counter+=1;
        if(sample_counter> year_sample):
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


# ## remove stop words
tokenized_docs_no_stopwords = tokenized_docs_no_punctuation;
# for doc in tokenized_docs_no_punctuation:
#     new_term_vector = []
#     for word in doc:
#         if not word in stopwords.words('english'):
#             new_term_vector.append(word)
#     tokenized_docs_no_stopwords.append(new_term_vector)
#
# print(tokenized_docs_no_stopwords);


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
file = open(out_dir+'FT_processed_10k_time_sample.p', 'wb')
pickle.dump([preprocessed_docs, year_label], file);

file = open(out_dir+'FT_raw_10k_time_sample.p', 'wb')
pickle.dump([raw_preprocessed, year_label], file);
