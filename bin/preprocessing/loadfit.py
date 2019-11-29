from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import os
import numpy as np
def loading(HEIGHT = 300, WIDTH = 300,
BATCH_SIZE = 8, VALIDATION_SPLIT = 0.2,
ROT_RANGE = 90, H_FLIP = True, SHEAR_RANGE = 0.2,
ZOOM_RANGE = 0.2, V_FLIP = True, ):

    TRAIN_DIR = os.getcwd() + "/dataset/Collapsed/samples/"
    train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    #featurewise_center = True,
    #samplewise_center = True,
    #zca_whitening = True,
    rotation_range=ROT_RANGE,
    horizontal_flip=H_FLIP,
    shear_range=SHEAR_RANGE,
    zoom_range =ZOOM_RANGE,
    vertical_flip=V_FLIP,
    #validation_split=VALIDATION_SPLIT
    )
    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=1,
                                                    #subset = 'training',
                                                    shuffle = True)
    #validation_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        #target_size = (HEIGHT, WIDTH),
                                                        #batch_size=BATCH_SIZE,
                                                        #subset = 'validation',
                                                        #shuffle = True)
    train_x = np.vstack(next(train_generator)[0] for _ in range(1))

    train_actual = ImageDataGenerator(rescale = 1. /255.,
                                      zca_whitening = True,
                                      validation_split = VALIDATION_SPLIT,
                                      )

    train_actual.fit(train_x)

    train_final = train_actual.flow_from_directory(TRAIN_DIR,
                                                   subset = 'training',
                                                   batch_size = BATCH_SIZE,
                                                   shuffle = True)
    validation_final = train_actual.flow_from_directory(TRAIN_DIR,
                                                        batch_size = BATCH_SIZE,
                                                        shuffle = True,
                                                        subset = 'validation')
    return train_final, validation_final
if __name__ == '__main__':
    loading()
