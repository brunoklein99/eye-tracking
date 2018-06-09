import cv2
from time import sleep
from keyboard import is_pressed
from pyautogui import position, size


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    w, h = size()
    i = 0
    with open('data/metadata.csv', 'w') as f:
        f.write('imagename,x,y\n')
        while True:
            if is_pressed('q'):
                break
            if is_pressed('r'):

                _, frame = cap.read()
                x, y = position()
                x /= w
                y /= h

                xr = round(x, 3)
                yr = round(y, 3)

                filename = 'data/{}-{}-{}.jpg'.format(i, xr, yr)
                cv2.imwrite(filename, frame)
                f.write('{},{},{}\n'.format(filename, xr, yr))

                i += 1
                sleep(0.25)
    print('finished')
