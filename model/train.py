import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import colorful as cf
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import common

BATCH_SIZE = 128
MAX_EPOCHS = 240
PATIENCE = MAX_EPOCHS // 3

LOG_DIR = './logs/'

IMAGE_SHAPE = (common.INPUT_SHAPE[0], common.INPUT_SHAPE[1])


def main(batch_size, model_name):
    show_settings(batch_size)

    model = make_model()

    training, validation = make_generators(batch_size)

    fit(model, training, validation, batch_size)

    save(model, model_name)

    print("\nFinished!")
    print("Run " + cf.skyBlue("classify.py") + 
          " to test the model and get information about its performance.")
    print("More information available with " + cf.orange("Tensorboard") + ".")


def show_settings(batch_size):
    print("Classification model training application started.\n")

    settings = pd.DataFrame(index=('max_epochs', 'patience', 'batch_size'), 
                            columns=('value', ))

    settings['value']['max_epochs'] = MAX_EPOCHS
    settings['value']['patience'] = PATIENCE
    settings['value']['batch_size'] = batch_size

    print(cf.skyBlue("Settings"))
    print(settings)


def make_model():
    print("\nCreating model...")

    model = Sequential()

    # Convolution block 1
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=common.INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolution block 2
    model.add(Conv2D(48, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Convolution block 3
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(len(common.CLASSES), activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def make_generators(batch_size):
    print("\nCreating generators...")

    aug = ImageDataGenerator(
        width_shift_range=0.125, height_shift_range=0.125, zoom_range=0.2)

    training = aug.flow_from_directory(
        common.TRAINING_DIR,
        target_size=IMAGE_SHAPE,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    validation = aug.flow_from_directory(
        common.VALIDATION_DIR,
        target_size=IMAGE_SHAPE,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical')

    return training, validation


def fit(model, training, validation, batch_size):
    print("\nFitting model...")

    history = model.fit_generator(
        training,
        epochs=MAX_EPOCHS,
        validation_data=validation,
        steps_per_epoch=training.n // batch_size,
        validation_steps=validation.n // batch_size,
        callbacks=setup_callbacks(),
        workers=2,
        verbose=2)

    best_epoch = np.argmin(history.history['val_loss']) + 1
    print("\n" + cf.lightGreen("Best epoch: {}".format(best_epoch)))


def setup_callbacks():
    log_dir = LOG_DIR + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    early_stopping = EarlyStopping(patience=PATIENCE, verbose=1, restore_best_weights=True)

    return [tensorboard_callback, early_stopping]


def save(model, model_name):
    print("\nSaving model...")

    path = common.MODEL_DIR + model_name

    model.save(path)
    print("Model saved to " + cf.skyBlue(path))


if __name__ == "__main__":
    os.system('color')

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE,
                      help="Specifies the batch size")
    parser.add_argument('-m', '--model', type=str, default="arrow_model.h5",
                      help="Specifies the output model name")

    args = parser.parse_args()

    main(args.batch_size, args.model)
