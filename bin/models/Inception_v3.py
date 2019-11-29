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

def inception_model(
        train, #train_generator input from preprocessing
        test, #Validation genetrato input from preprocessing
        VALIDATION_SPLIT = 0.3, #Validation split obtained
        BATCH_SIZE = 32, #Batch sizing chosen with 32 considered to save memory
        NUM_EPOCHS = 100, #Decided to
        NUM_CLASS = 8,
        opt = 'adam',
        DROPOUT = 0.5,
        CLASS_WEIGHTS = None,
        PATIENCE = 30
            ):
    num_train_img = (1-VALIDATION_SPLIT)*sum([len(files) for r, d, files in os.walk(train.directory)])
    num_val_img   = VALIDATION_SPLIT*sum([len(files) for r, d, files in os.walk(train.directory)])
    STEPS_PER_EPOCH = num_train_img/BATCH_SIZE
    VALIDATION_STEPS = num_train_img/BATCH_SIZE
    base_incepv3_model = app.InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=train.image_shape
        )
    model3 = Sequential()
    model3.add(base_incepv3_model)
    model3.add(Flatten())
    #model3.add(Dropout(0.3))
    model3.add(Dense(64, activation='relu'))
    #model3.add(Dropout(0.3))
    model3.add(Dense(8, activation='softmax'))
    model3.compile(
              loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    model3.summary()
    cb_early = EarlyStopping(
                    monitor='val_loss',
                    mode = 'min',
                    verbose=1,
                    patience = PATIENCE
                        )
    name = input("Enter name to store under /var/checkpoints")
    filepath="./var/checkpoints/" + name + "_model_weights.h5"
    cb_MC = ModelCheckpoint(
                filepath,
                monitor=["acc"],
                verbose=1,
                mode='max'
                    )
    callbacks_list = [cb_early, cb_MC]


    #debug
    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #run_metadata= tf.RunMetadata()




    print('Training model...')
    history = model3.fit_generator(
                                train,
                                steps_per_epoch = STEPS_PER_EPOCH,
                                validation_data = test,
                                validation_steps = VALIDATION_STEPS,
                                shuffle = True,
                                epochs = NUM_EPOCHS,
                                verbose = 1,
                                class_weight = CLASS_WEIGHTS,
                                callbacks = callbacks_list
                                    )
    label_map = (train.class_indices)
    model.evaluate(test)
    test.reset()
    y_pred = model.predict_generator(
                                    test,
                                    verbose = 1
                                    )
    y_pred = np.argmax(y_pred, axis = -1)
    y_true = test.classes[test.index_array]
    matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    return history, matrix, class_report
