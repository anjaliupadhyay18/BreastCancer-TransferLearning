import numpy as np
import pandas as pd
import tensorflow as tf
import os
import imageio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.keras import applications as app
from tensorflow.keras import optimizers, losses, activations, models

#from skimage import io
import matplotlib.pyplot as plt

def CNN_model(
        train, test, VALIDATION_SPLIT = 0.2,
        BATCH_SIZE = 32, NUM_EPOCHS = 20, NUM_CLASS = 8
            ):
    num_train_img = (1-VALIDATION_SPLIT)*sum([len(files) for r, d, files in os.walk(train.directory)])
    num_val_img   = VALIDATION_SPLIT*sum([len(files) for r, d, files in os.walk(train.directory)])
    STEPS_PER_EPOCH = num_train_img/BATCH_SIZE
    VALIDATION_STEPS = num_train_img/BATCH_SIZE
    model2 = Sequential()
    model2.add(Conv2D(
        32, (3, 3), activation='relu',
        input_shape=train.image_shape)
        )
    model2.add(MaxPooling2D((2, 2)))
    model2.add(
        Conv2D(64, (3, 3), activation='relu')
        )
    model2.add(Flatten())
    model2.add(Dense(64, activation='relu'))
    model2.add(Dense(8, activation='softmax'))
    model2.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
              metrics=['accuracy'])
    model2.summary()

    history2 = model2.fit_generator(train, steps_per_epoch=STEPS_PER_EPOCH,
                              validation_data=test, validation_steps=VALIDATION_STEPS,
                              shuffle=True, epochs=NUM_EPOCHS, verbose=True)
    return history2
