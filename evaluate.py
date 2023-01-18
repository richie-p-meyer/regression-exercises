#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score

import wrangle





def plot_residuals(y,yhat):
    '''
    y is the variable you're using to make your prediction
    yhat is the residual from your predictions
    '''
    import matplotlib.pyplot as plt
    res = yhat-y
    plt.scatter(y,res)
    plt.show()  



def regression_errors(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then outputs: the sse, ess, tss, mse, and rmse
    '''
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = math.sqrt(MSE)
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
    
    return MSE, SSE, RMSE, ESS, TSS




def regression_errors_print(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then outputs: the sse, ess, tss, mse, and rmse
    '''
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = math.sqrt(MSE)
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
        
    print(f''' 
        SSE: {SSE: .4f}
        ESS: {ESS: .4f}
        TSS: {TSS: .4f}
        MSE: {MSE: .4f}
        RMSE: {RMSE: .4f}
        ''')



