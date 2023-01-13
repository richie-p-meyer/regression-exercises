#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import pandas as pd
import numpy as np
import wrangle


# In[2]:


def split_data(df):
    '''
    Splits a df into a train, validate, and test set. Use 'train' to explore data.
    '''
    full = df
    train_validate, test = train_test_split(df, train_size =.8, random_state = 91)
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 91)
    return train, validate, test


# In[3]:


def scale_minmax(train,validate,test):
    '''
    Takes in train, validate, and test sets and returns the minmax scaled dfs
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train)
    train[train.columns] = scaler.transform(train[train.columns])
    validate[validate.columns] = scaler.transform(validate[validate.columns])
    test[test.columns] = scaler.fit(test[test.columns])
    
    return train, validate, test


# In[ ]:

def get_split():
    df = wrangle.wrangle_zillow()
    train, validate, test = split_data(df)
    return train, validate, test


