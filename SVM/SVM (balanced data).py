#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification using SVM (balanced data)

# ### Import libraries

# In[186]:


import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import os, shutil
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import svm, metrics
from sklearn.model_selection import learning_curve
from sklearn.linear_model import SGDClassifier
from sklearn.utils.multiclass import unique_labels


# ### Load data

# In[128]:


img_width = 224
img_height = 224
batch_size = 8

def loading(HEIGHT = img_height, WIDTH = img_width,
BATCH_SIZE = batch_size, VALIDATION_SPLIT = 0.3,
ROT_RANGE = 90, H_FLIP = True, SHEAR_RANGE = 0.2,
ZOOM_RANGE = 0.2, V_FLIP = True):

    TRAIN_DIR = "C:/Users/Abdullah Abid/Desktop/Project/BreakHis Data"
    train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=ROT_RANGE,
    horizontal_flip=H_FLIP,
    shear_range=SHEAR_RANGE,
    zoom_range =ZOOM_RANGE,
    vertical_flip=V_FLIP,
    validation_split=VALIDATION_SPLIT
    )

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    subset = 'training',
                                                    class_mode='sparse')
    validation_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                        target_size = (HEIGHT, WIDTH),
                                                        batch_size=BATCH_SIZE,
                                                        subset = 'validation',
                                                        class_mode='sparse')
    return train_generator, validation_generator


# In[129]:


get_ipython().run_cell_magic('time', '', 'train_generator, validation_generator = loading()')


# ### SVM classifier

# In[130]:


clf = SGDClassifier(loss="hinge", penalty="l2")


# In[131]:


get_ipython().run_cell_magic('time', '', '\ncount = 0\nfor inputs_batch, labels_batch in train_generator:\n    count = count + 1\n    X_train = []\n    y_train = []\n    for i in range(0, batch_size):\n        X_train.append(inputs_batch[i].flatten())\n        y_train.append(labels_batch[i])\n    X_train = np.array(X_train)\n    y_train = np.array(y_train)\n    y_train = y_train.astype(int)\n    clf.partial_fit(X_train, y_train, classes=np.unique(train_generator.classes))\n    if count * batch_size >= train_generator.n:\n        break')


# In[132]:


get_ipython().run_cell_magic('time', '', '\ny_test = []\ny_pred = []\ncount = 0\n\nfor inputs_batch, labels_batch in validation_generator:\n    count = count + 1\n    X_test = []\n    for i in range(0, batch_size):\n        X_test.append(inputs_batch[i].flatten())\n        y_test.append(labels_batch[i])\n    X_test = np.array(X_test)\n    y_pred.append(clf.predict(X_test))\n    if count * batch_size >= validation_generator.n:\n        break\n        \ny_test = np.array(y_test)\ny_test = y_test.astype(int)\ny_pred = np.array(y_pred)\ny_pred = y_pred.flatten()')


# ### Classification report

# In[133]:


print(metrics.classification_report(y_test, y_pred))


# ### Confusion matrix

# In[187]:


# code reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[188]:


class_names = []

for key in train_generator.class_indices.keys():
    class_names.append(key)

class_names = np.array(class_names)


# In[195]:


# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[193]:


# Plot normalized confusion matrix
np.set_printoptions(precision=2)
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()

