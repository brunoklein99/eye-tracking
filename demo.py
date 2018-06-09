import cv2
import numpy as np
import preprocess
import pandas as pd
from time import sleep
from keyboard import is_pressed
from pyautogui import position, size, moveTo

from model import get_model

if __name__ == '__main__':
    model = get_model()
    model.load_weights('weights/weights.hdf5')

    cap = cv2.VideoCapture(0)

    w, h = size()

    face_cascade, eye_cascade = preprocess.get_cascades()

    # df = pd.read_csv('data/metadata.csv')
    # for imagename in df['imagename']:
    while True:
        try:
            _, img = cap.read()
            # img = cv2.imread(imagename)

            face, eye1, eye2 = preprocess.get_eyes_img(img, face_cascade, eye_cascade)
            eye1 = np.expand_dims(eye1, axis=0)
            eye1 = np.expand_dims(eye1, axis=-1)
            eye2 = np.expand_dims(eye2, axis=0)
            eye2 = np.expand_dims(eye2, axis=-1)

            #x = np.concatenate((eye1, eye2), axis=0)
            x = eye2
            x = x.astype(dtype=np.float32)
            x /= 255
            x -= 0.4825
            x /= 0.164826

            y_hat = model.predict(x)
            x_pos = int(y_hat[0, 0] * w)
            y_pos = int(y_hat[0, 1] * h)

            moveTo(x_pos, y_pos)

            print(y_hat)
            sleep(1)
        except:
            print('exception')
