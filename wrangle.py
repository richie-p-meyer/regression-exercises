#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import env


# In[7]:


def get_zillow_2017():   
    if os.path.exists('zillow_2017.csv'):
        return pd.read_csv('zillow_2017.csv',index_col=0)
    else:
        url = env.get_connection('zillow')
        query = 'select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,         taxamount, fips  from properties_2017 join propertylandusetype using (propertylandusetypeid) where         propertylandusedesc = "Single Family Residential"'
        df = pd.read_sql(query,url)
        df.to_csv('zillow_2017.csv')
        return df


# In[8]:


df = get_zillow_2017()


# In[11]:


# 2.1 million rows, 7 columns
df.shape


# In[13]:


# FIPS (a unique county identifier code), all columns should be numeric
df.head()


# taxvaluedollarcnt: total tax assessed value of the parcel  
# taxamount: total property tax assessed yearly

# In[15]:


# All dtypes are numeric - that's good
df.dtypes


# In[20]:


# There are 2152863 rows - 9337 nan values is less than .5% of the data
df.isnull().sum()


# In[27]:


# If we drop all nan values we drop from 2152863 to 2140235 = keeping 99.4% of data - good enough!
df.dropna().shape


# In[29]:


df = df.dropna()


# In[35]:


#Looking for errors in data - 18 bedrooms and 0 bathrooms?
df[df.bedroomcnt>15]


# In[37]:


# 5326 rows have either 0 bedrooms or bathrooms - again, such a small amount that it's worth dropping
# These are single family residential, so not looking at land without structures or a barn
df[(df.bedroomcnt==0) | (df.bathroomcnt==0)]


# In[40]:


# Drop all rows with either no bedrooms or bathrooms
df = df.drop(df[(df.bedroomcnt==0) | (df.bathroomcnt==0)].index)


# In[44]:


# Drop all properties under 250 square feet (this is a small tiny home)
df = df.drop(df[df.calculatedfinishedsquarefeet<250].index)


# In[46]:


# Drop all properties where value is less than taxes
df = df.drop(df[df.taxvaluedollarcnt<df.taxamount].index)


# In[49]:


# Drop all properties built before 1850 - suspicious
df = df.drop(df[df.yearbuilt<1850].index)


# In[52]:


# Drop properties that have more bathrooms than bedrooms
df = df.drop(df[df.bedroomcnt<df.bathroomcnt].index)


# In[53]:


df.describe()


# In[56]:


df.bedroomcnt.min()


# In[58]:


def wrangle_zillow():
    import prepare
    df = get_zillow_2017()

    # Lose very small amount dropping nan values
    df = df.dropna()
    # Drop all rows with either no bedrooms or bathrooms
    df = df.drop(df[(df.bedroomcnt==0) | (df.bathroomcnt==0)].index)
    # Drop all properties under 250 square feet (this is a small tiny home)
    df = df.drop(df[df.calculatedfinishedsquarefeet<250].index)
    # Drop all properties where value is less than taxes
    df = df.drop(df[df.taxvaluedollarcnt<df.taxamount].index)
    # Drop all properties built before 1850 - suspicious
    df = df.drop(df[df.yearbuilt<1850].index)
    # Drop properties that have more bathrooms than bedrooms
    df = df.drop(df[df.bedroomcnt<df.bathroomcnt].index)
    
    train, validate, test = prepare.split_data(df)
    
    return df, train, validate, test


# # Based on the work you've done, choose a scaling method for your dataset. Write a function within your prepare.py that accepts as input the train, validate, and test data splits, and returns the scaled versions of each. Be sure to only learn the parameters for scaling from your training data!

# In[ ]:




