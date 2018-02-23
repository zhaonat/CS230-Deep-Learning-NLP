import numpy as np
import os
# sort files

dir = 'D:\\Downloads\\'

for file in os.listdir(dir):
    if(file.endswith('.json')):
        os.remove(dir+file);

