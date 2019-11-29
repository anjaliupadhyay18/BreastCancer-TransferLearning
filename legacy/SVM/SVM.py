#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Classification using SVM

# ### Import libraries

# In[1]:


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


# ### Function to load images from structured directory

# In[2]:


# code reference: https://github.com/whimian/SVM-Image-Classification/blob/master/Image%20Classification%20using%20scikit-learn.ipynb

def load_image_files(container_path, dimension=(150, 150)):
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

# In[5]:


clf = svm.SVC(gamma='scale')


# In[6]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X_train, y_train)')


# In[7]:


clf.score(X_test, y_test)


# In[8]:


get_ipython().run_cell_magic('time', '', 'y_pred = clf.predict(X_test)')


# ### Classification report

# In[9]:


print(metrics.classification_report(y_test, y_pred))


# In[10]:


image_dataset.target_names


# # Multiclass Classification (A, DC, F, LC, MC, PC, PT or TA)

# ### Load data

# In[3]:


get_ipython().run_cell_magic('time', '', 'image_dataset = load_image_files("C:/Users/Abdullah Abid/Desktop/Project/BreakHis Data Subclasses/")')


# ### Train-test split

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(image_dataset.data, image_dataset.target, test_size=0.3, random_state=100)


# ### SVM Classifier

# In[5]:


clf = svm.SVC(gamma='scale')


# In[6]:


get_ipython().run_cell_magic('time', '', 'clf.fit(X_train, y_train)')


# In[7]:


clf.score(X_test, y_test)


# In[8]:


get_ipython().run_cell_magic('time', '', 'y_pred = clf.predict(X_test)')


# ### Classification report

# In[9]:


print(metrics.classification_report(y_test, y_pred))


# In[12]:


image_dataset.target_names

