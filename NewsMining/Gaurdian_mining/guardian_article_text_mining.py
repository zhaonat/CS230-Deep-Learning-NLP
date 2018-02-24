
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import settings
import pandas as pd
import re
import string
import pickle
import time

regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)


dir = settings.ROOT_DIR+'\\Gaurdian\\'

file = dir+'Guardian_10K_epoch_1.csv';

data = pd.read_csv(file, index_col = 0);
print(data.shape)
print(data.columns);

document_data = list();
no_text_counter = 0; doc_counter = 0; #doc_counter so we can restrict samples
section_label = list();
num_examples = 10000;
for article in data.T.columns:
    print(doc_counter)
    if(doc_counter > num_examples):
        break;
    text = (data.T[article]['text'])
    if(isinstance(text, str)):
        document_data.append(text);
        section_label.append(data.T[article]['sectionId'])
        doc_counter += 1
    else:
        print(str(text) +', ' +str(no_text_counter));
        no_text_counter+=1;

print('failed entries: '+str(no_text_counter));
## now execute processor
## tokenization step
print('begin processing')

tokenized_docs = [word_tokenize(doc) for doc in document_data]
print(tokenized_docs)

## remove punctuation
tokenized_docs_no_punctuation = []
start = time.time()
punc_count = 0;
for review in tokenized_docs:
    new_review = []
    for token in review:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            new_review.append(new_token)
    punc_count +=1;
    print(punc_count);
    tokenized_docs_no_punctuation.append(new_review)

print(tokenized_docs_no_punctuation);
end = time.time()
print(end-start)

start = time.time()
## remove stop words: THIS IS THE MOST EXPENSIVE PART cuz it is O(n^2)
tokenized_docs_no_stopwords = []
for doc in tokenized_docs_no_punctuation:
    new_term_vector = []
    for word in doc:
        if not word in stopwords.words('english'):
            new_term_vector.append(word)
    tokenized_docs_no_stopwords.append(new_term_vector)

print(tokenized_docs_no_stopwords);
end = time.time();
print(end-start);

## stemming and lemmatizing
porter = PorterStemmer()
snowball = SnowballStemmer('english')
wordnet = WordNetLemmatizer()

preprocessed_docs = tokenized_docs_no_stopwords;
# #tokenized_docs_no_stopwords = tokenized_docs_no_punctuation;
# for doc in tokenized_docs_no_stopwords:
#     final_doc = []
#     for word in doc:
#         final_doc.append(porter.stem(word))
#         #final_doc.append(snowball.stem(word))
#         #final_doc.append(wordnet.lemmatize(word)) #note that lemmatize() can also takes part of speech as an argument!
#     preprocessed_docs.append(final_doc)

print(preprocessed_docs)

## convert preprocessed doc back into sentences

raw_preprocessed = [' '.join(x) for x in preprocessed_docs];


out_dir = settings.ROOT_DIR+'\\processed_data\\'
## save the corpus (just a list of lists of words in the document
file = open(out_dir+'Guardian_epoch_1_with_labels_no_lemma.p', 'wb')
pickle.dump([preprocessed_docs, section_label], file);

file = open(out_dir+'Guardian_raw_epoch_1_with_labels_no_lemma.p', 'wb')
pickle.dump([raw_preprocessed, section_label], file);
