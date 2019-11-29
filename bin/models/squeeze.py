#import base packages
import argparse
from time import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import ConfigParser
import io

#import necessary keras modules
from keras.squeezenet import SqueezeNet
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Conv2D
from keras.utils import plot_model
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.imagenet_utils import preprocess_input

def squeeze_model(
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
    base_model = SqueezeNet(
                    include_top = False,
                    weights = 'imagenet'
                    )
    # We add final layers
    x = base_model.output
    x = Dropout(rate = DROPOUT)(x)
    x = Conv2D(8, kernel_size = (1, 1), strides = (1,1), activation = 'relu')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASS, activation = 'softmax')(x)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

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
                mode='max')
    callbacks_list = [cb_early, cb_MC]
    model = Model(
                inputs=base_model.input,
                outputs=predictions
                    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=opt,
        metrics=["accuracy"]
            )

    history = model.fit_generator(
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
