import settings
from keras.optimizers import SGD

from data import get_data
from model import get_model

if __name__ == '__main__':
    model = get_model()
    model.summary()

    opt = SGD(lr=0.01)

    model.compile(loss='mae', optimizer=opt)

    x, y = get_data()

    model.fit(x, y,
              batch_size=settings.BATCH_SIZE,
              epochs=settings.EPOCHS,
              verbose=1)