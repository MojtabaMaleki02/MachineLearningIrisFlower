#!/usr/bin/env python
# coding: utf-8

# In[8]:


import matplotlib
print("matlotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("numpy version: {}".format(np.__version__))
import scipy as sp 
print("scipy version: {}".format(sp.__version__))
import IPython 
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))
from sklearn.datasets import load_iris
iris_dataset = load_iris()
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
from sklearn.neighbors import KNeighborsClassifier


# In[9]:


print ("key of iris_dataset:\n{}".format(iris_dataset.keys()))


# In[10]:


print(iris_dataset['DESCR'][:193]+"\n...")


# In[11]:


print("Target names:{}".format(iris_dataset['target_names']))


# In[12]:


print("Feature names:\n{}".format(iris_dataset['feature_names']))


# In[13]:


print("Type of data: {}".format(type(iris_dataset['data'])))


# In[14]:


print("Shape of data:{}".format(iris_dataset['data'].shape))


# In[15]:


print("first five rows of data:\n{}".format(iris_dataset['data'][:5]))


# In[16]:


print("Type of target:{}".format(type(iris_dataset['target'])))


# In[17]:


print("Shape of target:{}".format(iris_dataset['target'].shape))


# In[18]:


print("Target:\n{}".format(iris_dataset['target']))


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)


# In[20]:


print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))


# In[21]:


print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))


# In[22]:


iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[26]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


# In[27]:


X_new=np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape:{}".format(X_new.shape))


# In[28]:


prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("predicted target name: {}".format(iris_dataset['target_names'][prediction]))


# In[29]:


y_pred=knn.predict(X_test)
print("Test set prediction:\n{}".format(y_pred))


# In[30]:


print("Test score: {:.2f}".format(np.mean(y_pred==y_test)))


# In[31]:


print("Test set score:{:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:





# In[ ]:




