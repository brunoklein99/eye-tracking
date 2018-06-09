import settings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import seed
from keras.optimizers import SGD

from data import get_data
from model import get_model

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    seed(0)

    model = get_model()
    model.summary()

    opt = SGD(lr=0.01)

    model.compile(loss='mae', optimizer=opt)

    x_train, y_train, x_valid, y_valid = get_data()

    hist = model.fit(x_train, y_train,
                     validation_data=(x_valid, y_valid),
                     batch_size=settings.BATCH_SIZE,
                     epochs=settings.EPOCHS,
                     verbose=1)

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(range(len(loss)), loss, color='r')
    plt.plot(range(len(val_loss)), val_loss, color='g')
    plt.legend(['loss', 'val_loss'])
    plt.show()

    print('')