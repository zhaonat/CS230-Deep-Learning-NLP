import numpy as np

import pickle
import settings

file = settings.ROOT_DIR + '\\processed_data\\Guardian_epoch_1_with_labels.p';

[data, labels] = pickle.load(open(file, 'rb'));

print(set(labels))
print(len(set(labels)))

## label histogram:
label_dict = dict();
label_counts = dict();
c = 0;
for label in labels:
    if(label not in label_dict.keys()):
        label_dict[label] = list();
        label_counts[label] = 0;
    label_dict[label].append(data[c]);
    label_counts[label]+=1;
    c+=1;

other = list();
for label in label_counts.keys():
    if(label_counts[label] < 50):
        for article in label_dict[label]:

            other.append(article);
        label_dict.pop(label, None)
label_dict['other'] = other;
print(label_dict.keys());

#re-expand the label_dict back into a data and labels
new_data = list();
new_labels = list();
for key in label_dict.keys():
    articles = label_dict[key];
    for article in articles:
        new_data.append(' '.join(word for word in article))

    label = [key]*len(articles);
    new_labels = new_labels+label;
    #new_data.append(articles);



pickle.dump([new_data, new_labels], open('condensed_label_guardian_dataset.p', 'wb'));