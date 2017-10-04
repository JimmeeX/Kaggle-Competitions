
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[2]:


data = pd.read_csv("train.csv", index_col = "PassengerId")
display(data.head())


# In[3]:


data.pop("Name")
data.pop("Ticket")
display(data.head())


# In[5]:


data.info()


# In[6]:


x = data.fillna(data.median())
x.info()


# In[7]:


x["Cabin"] = x["Cabin"].fillna("Missing")
x.info()


# In[8]:


x = x.fillna(x.mode().iloc[0])
x.info()

