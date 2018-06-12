import settings
from keras import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, concatenate


def get_model_multinput():
    in1 = Input(shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1))
    in2 = Input(shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1))
    cv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1))
    cv2 = Conv2D(64, (3, 3), activation='relu')
    mp1 = MaxPooling2D(pool_size=(2, 2))
    flt = Flatten()

    x1 = cv1(in1)
    x1 = cv2(x1)
    x1 = mp1(x1)
    x1 = flt(x1)

    x2 = cv1(in2)
    x2 = cv2(x2)
    x2 = mp1(x2)
    x2 = flt(x2)

    x = concatenate([x1, x2])
    x = Dense(128, activation='relu')(x)
    x = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[in1, in2], outputs=x)

    return model


def get_model():
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(settings.IMAGE_SIZE, settings.IMAGE_SIZE, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    return model
