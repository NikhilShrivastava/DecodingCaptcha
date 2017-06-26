from scipy.misc import imread

import pandas as pd


def gray_to_csv(rec_file):
    img = imread(rec_file,0)
    # cv2.imshow("Current Single file", img)
    # cv2.waitKey(10000)
    value = img.flatten()
    df = pd.DataFrame(value).T
    df = df.sample(frac=1)  # shuffle the dataset
    with open('single.csv', 'w') as dataset:
        df.to_csv(dataset, header=False, index=False)
    return 'single.csv'
