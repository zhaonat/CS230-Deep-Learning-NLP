import nltk
import os
import json
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pickle
from settings import ROOT_DIR
import re
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

for file in os.listdir(data_dir):

    data = json.load(open(data_dir+file,  encoding="utf8"));
    #pprint(data)

    text = data['bodyXML']
    document_data.append(text);

    #need a more advanced function to parse this

    text = remove_tags(text);
    text = text.strip('\n')


    ## convert to sentences with tokenize
    sentences = nltk.sent_tokenize(text)
    #sentences = text.split('.');
    print(sentences)
    # split each sentence in sentences into a list of words
    sentence_list = list();
    for sentence in sentences:
        words = nltk.word_tokenize(sentence);
        words = [w.lower() for w in words]
        sentence_list.append(words);
    word_sentence_data.append(sentence_list);
    # create the transform
    print(word_sentence_data)
    counter+=1;
    if(counter> 100):
        break;

## save text_data
file = open(out_dir+'document_text.p', 'wb')
pickle.dump(document_data, file);
file2 = open(out_dir+'sentence_text.p', 'wb');
pickle.dump(word_sentence_data, file2);
