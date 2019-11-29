from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
TRAIN_DIR = "/dataset/Collapsed/samples/"
HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8
VALIDATION_SPLIT = 0.2
train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=90,
      horizontal_flip=True,
      shear_range=0.2,
      zoom_range =0.2,
      vertical_flip=True,
      validation_split=VALIDATION_SPLIT
    )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR, 
                                                    target_size=(HEIGHT, WIDTH), 
                                                    batch_size=BATCH_SIZE,
                                                       subset = 'training')
validation_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size = (HEIGHT, WIDTH),
                                                        batch_size=BATCH_SIZE,
                                                        subset = 'validation')
