import settings
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model
