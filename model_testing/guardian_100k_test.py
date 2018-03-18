import pickle
import settings
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

## load the guardian unlabelled deployment data
guardian = settings.ROOT_DIR+'\\Guardian_corpus\\Guardian_unlabelled_tokens_100k.p'
[tokenizer, sequences] = pickle.load(open(guardian, 'rb'));

## load the text;
dir = 'D:/Documents/Classes/CS230/Guardian_corpus/'

## open one of the processed data sets so we can extract relevant label info
dir2 = 'D:/Documents/Classes/CS230/processed_data/'

guardian = dir2+'Guardian_raw_epoch_1_with_labels.p';
#guardian = dir+'Guardian_epoch_1_with_labels.p';

# process data to make it in the
[guardian_corpus, labels] = pickle.load(open(guardian, 'rb'));
count_label = Counter(labels)
for i in range(len(guardian_corpus)):
    if Counter(labels)[labels[i]] < 50: #here we do the label condensation IN SCRIPT
        labels[i] = 'other'

#convert labels to one hot encoding
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
print(integer_encoded)
print('size of data: '+str(len(integer_encoded)));
number_to_name = dict();
integer_encoded = integer_encoded.reshape(len(integer_encoded), )
for i in range(len(integer_encoded)):
    if(integer_encoded[i] not in number_to_name.keys()):
        number_to_name[integer_encoded[i]] = labels[i];


#guardian = dir+'Guardian_epoch_1_with_labels.p';
guardian_corpus = pd.read_csv(dir+'filtered_guardian_unlabelled_100k.csv');
guardian_corpus = guardian_corpus['text'].values

## =========================
print(sequences)

max_features = 4000;
data = pad_sequences(sequences, maxlen=max_features)

dir = settings.ROOT_DIR+'\\saved_models\\'
model_name = dir+'guardian_CNN.h5';

model = load_model(model_name);
predicted_labels = np.argmax(model.predict(data, batch_size = 128, verbose = True), axis = 1);

pred_label_names = list(map(lambda x: number_to_name[x], predicted_labels))

predictions = pd.DataFrame(predicted_labels, columns = ['labels']);
predictions['label_names'] = pred_label_names;
predictions['texts'] = guardian_corpus;

predictions.sort_values(['labels'], inplace = True)

## we should split the final predictions to smaller csv files so we can actually open one
predictions.to_csv('guardian_final_predictions_100k.csv')







