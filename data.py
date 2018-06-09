import pandas as pd
import cv2
import numpy as np
import settings
from preprocess import get_eyes


def get_data():
    df = pd.read_csv('data/metadata.csv')

    eyes = get_eyes(df['imagename'].values)
    eyes = list(eyes)

    x = []
    y = []

    for i, (face, (x1, y1, w1, h1), (x2, y2, w2, h2)) in enumerate(eyes):
        img = face[y1:y1 + h1, x1:x1 + w1]
        img = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        x.append(img)

        img = face[y2:y2 + h2, x2:x2 + w2]
        img = cv2.resize(img, (settings.IMAGE_SIZE, settings.IMAGE_SIZE))
        x.append(img)

        row = df.iloc[i]
        x_pos = row['x']
        y_pos = row['y']

        y.append((x_pos, y_pos))
        y.append((x_pos, y_pos))

    x = np.array(x, dtype=np.float32)
    x = np.expand_dims(x, axis=-1)
    x /= 255
    x -= np.mean(x)
    x /= np.std(x)
    y = np.array(y, dtype=np.float32)

    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    return x, y
