from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
import os
def loading(HEIGHT = 224, WIDTH = 224,
BATCH_SIZE = 8, VALIDATION_SPLIT = 0.3,
ROT_RANGE = 90, H_FLIP = True, SHEAR_RANGE = 0.2,
ZOOM_RANGE = 0.2, V_FLIP = True):

	TRAIN_DIR = os.getcwd() + "/dataset/Collapsed/samples/"
	train_datagen =  ImageDataGenerator(
	rescale = 1./255,
	preprocessing_function=preprocess_input,
	#featurewise_center = True,
	#samplewise_center = True,
	#zca_whitening = True,
	rotation_range=ROT_RANGE,
	horizontal_flip=H_FLIP,
	shear_range=SHEAR_RANGE,
	zoom_range =ZOOM_RANGE,
    vertical_flip=V_FLIP,
    validation_split=VALIDATION_SPLIT,
    )

	train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    subset = 'training',
													shuffle = True,
													color_mode = "rgb",
													class_mode = "categorical")
	validation_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size = (HEIGHT, WIDTH),
                                                        batch_size=BATCH_SIZE,
                                                        subset = 'validation',
														shuffle = False,
														color_mode = 'rgb',
														class_mode = "categorical")
	return train_generator, validation_generator
if __name__ == '__main__':
	loading()
