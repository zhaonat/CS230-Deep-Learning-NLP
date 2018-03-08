import numpy as np
import pickle
import settings
import matplotlib.pyplot as plt

file = settings.ROOT_DIR + '\\processed_data\\Guardian_epoch_1_with_labels.p';

[data, labels] = pickle.load(open(file, 'rb'));

print(set(labels))
print(len(set(labels)))

lengths = list();
for doc in data:
    lengths.append(len(doc))

print(np.mean(lengths));
print(np.max(lengths));
print(np.min(lengths));
print(np.sqrt(np.var(lengths)))
plt.figure();
plt.hist(lengths, 30)
plt.show()
## label histogram:
label_dict = dict();
for label in labels:
    if(label not in label_dict.keys()):
        label_dict[label] = 0;
    label_dict[label]+=1;

print(label_dict);

counter = 0;
for i in label_dict.keys():
    if(label_dict[i] < 50):
        counter+=1;
    else:
        print(str(i)+', '+str(label_dict[i]));

print('num labels with less than 50 examples: '+str(counter))

## print sample
print(len(labels))
sample = np.random.randint(0, 1000, 10);
for i in sample:
    print(labels[i])
    print(data[i])


