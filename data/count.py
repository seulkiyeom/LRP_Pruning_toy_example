import numpy as np
import glob

npy_files = glob.glob('./*.npy')
for npy in npy_files:
    data =  np.load(npy)
    print('{}: {}'.format(npy, data.shape))