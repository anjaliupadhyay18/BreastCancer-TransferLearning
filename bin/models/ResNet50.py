import numpy as np
import pandas as pd
import tensorflow as tf
import os
import imageio
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, Activation, GlobalAveragePooling2D, MaxPooling2D, Conv2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from skimage import io
import matplotlib.pyplot as plt

def resnet50_model(
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
    VALIDATION_STEPS = num_val_img/BATCH_SIZE
    base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(train.image_shape),
                    pooling = 'avg'
                        )
    model4 = Sequential()
    model4.add(base_model)
    model4.add(Dense(512, activation = 'relu'))
    model4.add(Dropout(DROPOUT))
    model4.add(Dense(8, activation = 'softmax'))
    model4.compile(loss= 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    model4.summary()
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
    batch_size = 8
    history4 = model4.fit_generator(
                                train,
                                #class_weight = cw,
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
    model4.evaluate(test)
    test.reset()
    y_pred = model4.predict_generator(
                                    test,
                                    verbose = 1
                                    )
    y_pred = np.argmax(y_pred, axis = -1)
    y_true = test.classes[test.index_array]
    matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    return history4, matrix, class_report
