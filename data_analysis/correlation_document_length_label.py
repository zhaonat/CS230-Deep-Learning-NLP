
import pickle
import settings
import matplotlib.pyplot as plt
import numpy as np

dir = settings.ROOT_DIR+'\\processed_data\\'
file = 'Econ_corpus_raw.p'

[data, labels] = pickle.load(open(dir+file, 'rb'));

doc_lengths = list();
doc_length_mapper = dict();
c = 0;
for doc in data:
    words = doc.split(' ');
    doc_lengths.append(len(words));
    lab = labels[c];
    if(lab not in doc_length_mapper.keys()):
        doc_length_mapper[lab] = list();
    doc_length_mapper[lab].append(len(words))
    c+=1;
    if(c%1000 == 0):
        print(c)
print(np.mean(doc_lengths));
for key in doc_length_mapper:
    print(key+', '+str(np.mean(doc_length_mapper[key]))+', num_docs: '+str(len(doc_length_mapper[key])))

## correlate misclassifications to the longest articles

