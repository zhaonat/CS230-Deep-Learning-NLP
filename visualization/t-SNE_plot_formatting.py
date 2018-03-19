import settings
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

SMALL_SIZE = 20
MEDIUM_SIZE = 50
BIGGER_SIZE = 100
plt.rc('axes', linewidth=2)
plt.rc('axes', )
plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=28)  # fontsize of the tick labels
plt.rc('ytick', labelsize=28)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

data_dir = os.path.join(settings.ROOT_DIR, 'visualization')
data = pd.read_csv(os.path.join('guardian_vs_econ_t-SNE.csv'), index_col = 0)
print(data);
data = data.values;
sample = np.random.randint(0, len(data[:,0]), 30000)
X = data[sample,0:2];
y = data[sample,2];
plt.figure(figsize = (40,30))
plt.scatter(X[:,0], X[:,1], c = y, edgecolor='black', s = 420);
plt.set_cmap('jet')
#plt.colorbar()
plt.title('t-SNE visualization of TOPIC DISTRIBUTION of the Economist', fontsize = 60)

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off', right='off', left='off', labelleft='off') # labels along the bottom edge are off

class_colours = ['r', 'b']
import matplotlib.patches as mpatches
recs = list(); classes = ['economist','guardian']
for i in range(0,len(class_colours)):
    recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
plt.legend(recs,classes,loc=4, fontsize = 50)

plt.xlabel('embedding dimension 1', fontsize = 50);
plt.ylabel('embedding dimension 2', fontsize = 50)
plt.savefig('economist_guardian_t-SNE.png')
plt.show()


