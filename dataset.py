from scipy.misc import imread
import numpy as np
import os
import pandas as pd

for directory, subdirectories, files in os.walk("./captcha-test"):  # for test: ./cat-test
    for file in files:
        im = imread(os.path.join(directory, file))
        value = im.flatten()
        value = np.hstack((directory[12:], value))

        df = pd.DataFrame(value).T
        with open('test_initial.csv', 'a') as f:  # for test: test_initial.csv
            df.to_csv(f, header=False, index=False)

df = pd.read_csv('test_initial.csv')  # for test: test_initial.csv
df = df.sample(frac=1)
df.to_csv('test.csv', header=False, index=False)  # for test: test.csv