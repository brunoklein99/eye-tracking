import cv2
import numpy as np
import pandas as pd

from data import get_data

if __name__ == '__main__':
    x, y = get_data()
    cv2.imshow('', x[0, :, :, 0])
    cv2.waitKey()
    print()