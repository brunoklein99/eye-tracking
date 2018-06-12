from random import shuffle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import get_eyes


def _eyes_to_arrays(eyes):
    x = []
    y = []
    for ((face, eyes), x_pos, y_pos) in eyes:
        x.append(eyes)
        y.append((x_pos, y_pos))

    x = np.array(x, dtype=np.float32)
    x /= 255
    y = np.array(y, dtype=np.float32)

    return x, y


def get_data():
    df = pd.read_csv('data/metadata.csv')

    eyes = get_eyes(df['imagename'].values)
    eyes = zip(eyes, df['x'], df['y'])
    eyes = list(eyes)
    shuffle(eyes)

    eyes_train, eyes_valid = train_test_split(eyes, test_size=0.3)

    train_x, train_y = _eyes_to_arrays(eyes_train)

    train_mean = np.mean(train_x)
    train_std = np.std(train_x)
    print('mean: ', train_mean)
    print('std: ', train_std)

    train_x -= train_mean
    train_x /= train_std

    valid_x, valid_y = _eyes_to_arrays(eyes_valid)

    valid_x -= train_mean
    valid_x /= train_std

    return train_x, train_y, valid_x, valid_y
