import settings
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

SMALL_SIZE = 50
MEDIUM_SIZE = 50
BIGGER_SIZE = 100
plt.rc('axes', linewidth=5)
plt.rc('font', size=50)  # cotrols default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=28)  # fontsize of the tick labels
plt.rc('ytick', labelsize=28)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=50)  # fontsize of the figure title



path = os.path.join(settings.ROOT_DIR, 'model_testing', 'guardian_final_predictions_100k.csv' )

data = pd.read_csv(path, index_col = 0);

## clean this data
data.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)

print(data.columns)

## get xticks...set does  not preserve order
xticks = data.label_names.unique();


#xticks = list(set(data['label_names']))
print(xticks)
plt.figure(figsize = (30,10))
numeric_labels = list(data['labels'].values);
#[hist_values, bin_edge] = np.histogram(numeric_labels);
plt.hist(numeric_labels, 27, edgecolor='black', linewidth=5, color = "firebrick", normed = True)
x = list(range(27))
plt.xticks(x, xticks);
plt.xticks(rotation=90)
plt.grid()
plt.title('News Focus of the Gaurdian Predicted by CNN')
plt.ylabel('Number of Articles')
plt.gcf().subplots_adjust(bottom=0.30)




## get distributoin of training articles from the guardian
path = os.path.join(settings.ROOT_DIR, 'Guardian_corpus', 'condensed_label_guardian_dataset.p');
[training_corpus, labels]  = pickle.load(open(path, 'rb'))

label_hist = dict();

for i in labels:
    if(i not in label_hist.keys()):
        label_hist[i] = 0;
    label_hist[i]+=1;
print(label_hist)
plt.hist(labels, 27, normed = True, edgecolor='black', linewidth=5)

#plt.show()

training_labels = le.fit_transform(labels)

all_labels = [training_labels, np.array(numeric_labels)]
plt.figure()
plt.hist(all_labels, 27,  normed = True,  edgecolor='black', linewidth=2, color = ['firebrick', 'royalblue'])
plt.legend(('train/test', 'deployment'))
plt.ylabel('proportion of articles')
x = list(range(27))
plt.xticks(x, xticks);
plt.xticks(rotation=80)
plt.gcf().subplots_adjust(bottom=0.3)

plt.savefig('Distribution of Predicted Articles vs Original.png')
plt.show()