#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification using SVM

# ### Import libraries

# In[12]:


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'notebook')
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split
import skimage.io
from skimage.transform import resize
from sklearn.utils.multiclass import unique_labels


# ### Function to load images from structured directory

# In[2]:


# code reference: https://github.com/whimian/SVM-Image-Classification/blob/master/Image%20Classification%20using%20scikit-learn.ipynb

def load_image_files(container_path, dimension=(224, 224)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images)


# ## Binary Classification (Benign or Malignant)

# ### Load data

# In[3]:


get_ipython().run_cell_magic('time', '', 'image_dataset = load_image_files("C:/Users/Abdullah Abid/Desktop/Project/BreakHis Data Classes/")')


# ### Train-test split

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=100)


# ### SVM Classifier

# In[6]:


get_ipython().run_cell_magic('time', '', "\nparams = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100]},\n          {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},\n          {'kernel': ['poly'], 'C': [0.01, 0.1, 1, 10, 100]}]\n\nsvc = svm.SVC(gamma='scale')\nclf = GridSearchCV(svc, params, iid='False', cv=5)\nclf.fit(X_train, y_train)")


# In[ ]:


clf.best_params_


# In[7]:


get_ipython().run_cell_magic('time', '', 'y_pred = clf.predict(X_test)')


# ### Classification report

# In[8]:


print(metrics.classification_report(y_test, y_pred))


# In[9]:


image_dataset.target_names


# ### Confusion matrix

# In[14]:


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


# In[15]:


class_names = np.array(image_dataset.target_names)


# In[16]:


# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix')
plt.show()


# # Multiclass Classification (A, DC, F, LC, MC, PC, PT or TA)

# ### Load data

# In[17]:


get_ipython().run_cell_magic('time', '', 'image_dataset = load_image_files("C:/Users/Abdullah Abid/Desktop/Project/BreakHis Data Subclasses/")')


# ### Train-test split

# In[18]:


X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=100)


# ### SVM Classifier

# In[20]:


get_ipython().run_cell_magic('time', '', "\nparams = [{'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100]},\n          {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},\n          {'kernel': ['poly'], 'C': [0.01, 0.1, 1, 10, 100]}]\n\nsvc = svm.SVC(gamma='scale', decision_function_shape='ovo')\nclf = GridSearchCV(svc, params, iid='False', cv=5)\nclf.fit(X_train, y_train)")


# In[ ]:


clf.best_params_


# In[21]:


get_ipython().run_cell_magic('time', '', 'y_pred = clf.predict(X_test)')


# ### Classification report

# In[22]:


print(metrics.classification_report(y_test, y_pred))


# In[23]:


image_dataset.target_names


# ### Confusion matrix

# In[24]:


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


# In[25]:


class_names = np.array(image_dataset.target_names)


# In[26]:


# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names,
                      title='Confusion matrix')
plt.show()

