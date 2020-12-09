import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import (Dense, Conv2D, MaxPooling2D,
                          Flatten, LeakyReLU, SpatialDropout2D, Dropout,
                          BatchNormalization)


def ImiCarla(input_shape, quiet=False):
    model = Sequential()

    model.add(
        Conv2D(32,
               kernel_size=7,
               padding="same",
               kernel_initializer="glorot_normal",
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(64,
               kernel_size=5,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(128,
               kernel_size=3,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(256,
               kernel_size=3,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(
        Conv2D(512,
               kernel_size=3,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(512, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation="linear"))

    if not quiet:
        model.summary()

    return model


def ImiSteer(input_shape, quiet=False):
    model = Sequential()

    model.add(
        Conv2D(32,
               kernel_size=7,
               padding="same",
               kernel_initializer="glorot_normal",
               input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(
        Conv2D(64,
               kernel_size=5,
               padding="same",
               kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())

    model.add(Dense(1024, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(512, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_initializer="glorot_normal"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation="sigmoid"))

    if not quiet:
        model.summary()

    return model
