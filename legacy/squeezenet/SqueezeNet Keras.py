from keras_squeezenet import SqueezeNet
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Activation, Conv2D
from keras.utils import plot_model
from keras.models import Model
from keras import optimizers 
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input
import argparse
from time import time
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# ### Initial parameters
# Specify path to training and testing data
# Data is in ImageNet folder structure
train_dir = 'C:/Users/Renee/Documents/2019/Semester 2/COMP5703/Data/breakhis/train'
val_dir   = 'C:/Users/Renee/Documents/2019/Semester 2/COMP5703/Data/breakhis/val'

num_train_img = sum([len(files) for r, d, files in os.walk(train_dir)])
num_val_img   = sum([len(files) for r, d, files in os.walk(val_dir)])

num_class  = 8
img_size = 224

batch_size = 32
num_epochs = 20

steps_per_epoch  = num_train_img/batch_size
validation_steps = num_val_img/batch_size


# ### Data

train_datagen = image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = image.ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size,img_size),
        batch_size=batch_size,
        class_mode='categorical')


# ### Pre-trained model
# Create a Squeezenet model without the last layers and load weights pre-trained on ImageNet data.  
# We can also change whether we want to train all parameters or just the final layers. We can achieve better results by retraining all parameters, but this also takes longer.

print('Loading pre-trained model...')

base_model = SqueezeNet(include_top=False, weights='imagenet')

# Print the layers of Squeezenet and retrainable parameters
i=0
for layer in base_model.layers:
    layer.trainable = True # True to retrain all layers 
    i = i+1
    print(i,layer.name)


# ### Fine tune 
# We add a final layer so that the model can predict on our own classes. We use a dropout layer of 20% and a softmax layer to return probabilities of class.

x = base_model.output
x = Dropout(rate = 0.5)(x)
x = Conv2D(8, kernel_size=(1, 1), strides=(1, 1), activation = 'relu')(x)
x = GlobalAveragePooling2D()(x)

predictions = Dense(num_class, activation='softmax')(x)


# ### Weights
# To deal with the class imbalance, we can add weights to the loss function to penalise heavier on rarer classes. There is no hard and fast rule to select the weights, but some commonly used weights are:
# * Number of total samples / number of samples per class
# * Number of samples of largest class / number of samples per class
# * 1 / number of samples per class

print('Calculating class weights...')

sample_count = [len(os.listdir(os.path.join(train_dir,tumour))) for tumour in os.listdir(train_dir)]
weights = [sum(sample_count)/count for count in sample_count]

class_weights = {i:val for i,val in enumerate(weights)}


# ### Run model

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

filepath = 'keras_squeezenet.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1,save_best_only=True,save_weights_only=False, mode='max',period=1)
callbacks_list = [checkpoint,tensorboard]


model = Model(inputs=base_model.input, outputs=predictions)

#debug
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata= tf.RunMetadata()

model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.01),metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.001, momentum=0.99),metrics=["accuracy"],options=run_options, run_metadata=run_metadata)
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(),metrics=["accuracy"])
#model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),metrics=["accuracy"])


print('Training model...')
history = model.fit_generator(
    train_generator,
    class_weight = class_weights,
    steps_per_epoch = steps_per_epoch,
    epochs = num_epochs,
    callbacks = callbacks_list,
    validation_data = validation_generator,
    validation_steps = validation_steps,
    workers = 4,
    max_queue_size = 500,
    shuffle = True,
    verbose = True
)


# Training and validation accuracy plot

plt.plot(range(num_epochs), history.history['accuracy'])
plt.plot(range(num_epochs), history.history['val_accuracy'])
plt.title("Accuracy of training and validation set")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Training set", "Validation set"])


# ### Confusion Matrix
model.evaluate(validation_generator)

validation_generator.reset()
y_pred = model.predict_generator(
            validation_generator,
            steps=validation_steps,
            verbose = 1)
y_pred = np.argmax(y_pred, axis=-1)

y_true = validation_generator.classes[validation_generator.index_array]

matrix = confusion_matrix(y_true,y_pred)
print(matrix)
print(np.diag(matrix)/sum(matrix))
print(sum(np.diag(matrix))/sum(sum(matrix)))

