
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

'''
    mine the 100,000 unlabelled articles into a corpus for future analysis and deployment, no LABELS, so remove that stuff
'''

regex = re.compile('[%s]' % re.escape(string.punctuation)) #see documentation here: http://docs.python.org/2/library/string.html

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)


dir = settings.ROOT_DIR+'\\Guardian_corpus\\'
file = dir+'Guardian_100K_epoch_4.csv';

data = pd.read_csv(file, index_col = 0);
print(data.shape)
print(data.columns);

document_data = list();
no_text_counter = 0; doc_counter = 0; #doc_counter so we can restrict samples
for article in data.T.columns:
    if(doc_counter%10000 == 0):
         print(doc_counter)

    text = (data.T[article]['text'])
    if(isinstance(text, str)):
        document_data.append(text);
        doc_counter += 1
    else:
        print(str(text) +', ' +str(no_text_counter));
        no_text_counter+=1;

print('failed entries: '+str(no_text_counter));
## now execute processor
## tokenization st

file = open(dir+'Guardian_unlabelled_corpus.p', 'wb')
pickle.dump([document_data], file);
