import settings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from random import seed
from keras.optimizers import SGD

from data import get_data
from model import get_model, get_model_multinput

if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    seed(0)

    model = get_model_multinput()
    model.summary()

    opt = SGD(lr=0.01, momentum=0.9, decay=1e-4)

    model.compile(loss='mae', optimizer=opt)

    x_train, y_train, x_valid, y_valid = get_data()

    x_train1 = np.expand_dims(x_train[:, :, :, 0], axis=-1)
    x_train2 = np.expand_dims(x_train[:, :, :, 1], axis=-1)
    x_valid1 = np.expand_dims(x_valid[:, :, :, 0], axis=-1)
    x_valid2 = np.expand_dims(x_valid[:, :, :, 1], axis=-1)

    hist = model.fit([x_train1, x_train2], y_train,
                     validation_data=([x_valid1, x_valid2], y_valid),
                     batch_size=settings.BATCH_SIZE,
                     epochs=settings.EPOCHS,
                     verbose=1)

    model.save_weights('weights/weights.hdf5', overwrite=True)

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']

    plt.plot(range(len(loss)), loss, color='r')
    plt.plot(range(len(val_loss)), val_loss, color='g')
    plt.legend(['loss', 'val_loss'])
    plt.show()

    print('')