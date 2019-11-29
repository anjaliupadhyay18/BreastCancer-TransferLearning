#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification using SVM and VGG feature extraction (balanced data)

# ### Import libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import os, shutil
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.utils.multiclass import unique_labels


# ### Instantiate convolutional base

# In[2]:


img_width = 224
img_height = 224

conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(img_width, img_height, 3))


# In[3]:


# Check architecture
conv_base.summary()


# ### Load data

# In[4]:


batch_size = 8

def loading(HEIGHT = img_height, WIDTH = img_width,
BATCH_SIZE = batch_size, VALIDATION_SPLIT = 0.3,
ROT_RANGE = 90, H_FLIP = True, SHEAR_RANGE = 0.2,
ZOOM_RANGE = 0.2, V_FLIP = True):

    TRAIN_DIR = "C:/Users/Abdullah Abid/Desktop/Project/BreakHis Data 2"
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


# In[5]:


get_ipython().run_cell_magic('time', '', 'train_generator, validation_generator = loading()')


# ### Extract features

# In[6]:


# code reference: https://github.com/pmarcelino/blog/blob/master/dogs_cats/dogs_cats.ipynb

def extract_features(sample_count, generator):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    
    return features, labels


# In[7]:


get_ipython().run_cell_magic('time', '', 'train_features, train_labels = extract_features(train_generator.n, train_generator)\nvalidation_features, validation_labels = extract_features(validation_generator.n, validation_generator)')


# ### SVM classifier

# In[8]:


get_ipython().run_cell_magic('time', '', "\nX_train, y_train = train_features.reshape(train_generator.n, 7*7*512), train_labels\nX_test, y_test = validation_features.reshape(validation_generator.n, 7*7*512), validation_labels\n\nparams = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100]},\n          {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},\n          {'kernel': ['poly'], 'C': [0.01, 0.1, 1, 10, 100]}]\n\nsvc = svm.SVC(gamma='scale', decision_function_shape='ovo')\nclf = GridSearchCV(svc, params, iid='False', cv=5)\nclf.fit(X_train, y_train)")


# In[7]:


clf.best_params_


# In[9]:


get_ipython().run_cell_magic('time', '', 'y_pred = clf.predict(X_test)')


# ### Classification report

# In[10]:


print(metrics.classification_report(y_test, y_pred))


# ### Confusion matrix

# In[11]:


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


# In[14]:


class_names = []

for key in train_generator.class_indices.keys():
    class_names.append(key)

class_names = np.array(class_names)
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)


# In[15]:


# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix')
plt.show()


# In[16]:


# Plot normalized confusion matrix
np.set_printoptions(precision=2)
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()


# ### Learning curve 

# In[11]:


get_ipython().run_cell_magic('time', '', '\nimport warnings\nwarnings.filterwarnings(\'ignore\')\n\n# code reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html\n\nplt.figure()\nplt.title("Learning Curve")\nplt.xlabel("Training samples")\nplt.ylabel("Score")\ntrain_sizes, train_scores, test_scores = learning_curve(estimator = clf,\n                                                        X = X_train,\n                                                        y = y_train)\ntrain_scores_mean = np.mean(train_scores, axis=1)\ntrain_scores_std = np.std(train_scores, axis=1)\ntest_scores_mean = np.mean(test_scores, axis=1)\ntest_scores_std = np.std(test_scores, axis=1)\nplt.grid()\n\nplt.fill_between(train_sizes, train_scores_mean - train_scores_std,\n                 train_scores_mean + train_scores_std, alpha=0.1,\n                 color="r")\nplt.fill_between(train_sizes, test_scores_mean - test_scores_std,\n                 test_scores_mean + test_scores_std, alpha=0.1, color="g")\nplt.plot(train_sizes, train_scores_mean, \'o-\', color="r",\n         label="Training score")\nplt.plot(train_sizes, test_scores_mean, \'o-\', color="g",\n         label="Validation score")\n\nplt.legend(loc="best")')

