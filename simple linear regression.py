#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[129]:


df = pd.read_csv("forestfires.csv")
df = df[['temp','area']]
df.head()


# In[82]:


## y = b_0 + b_1x
x = df.temp
y = df.area


# In[136]:


##visualize the data
plt.scatter(x,y,color='m')
plt.ylim(-1, 10)
plt.show()


# In[137]:


def estimate_coef(x,y):
    
    ## y = b_0 + b_1*x
    
    # number of observations/points 
    n = np.size(x)
    
    # mean of x and y vector 
    m_x, m_y = np.mean(x), np.mean(y) 
    
    # calculating cross-deviation and deviation about x 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x
    
    # calculating regression coefficients 
    b_1 = SS_xy / SS_xx 
    b_0 = m_y - b_1*m_x
    
    return(b_0,b_1)
print(estimate_coef(x,y))


# In[125]:


def correlation(x,y):
    ## for dataframe objects,both x amnd y are dataframe objects
    return(x.corr(y))

def standard_deviation(a):
    ## for dataframe objects
    return(a.std())

mean = lambda b:b.mean()

def least_squares_fit(x, y):
    """given training values for x and y,
    find the least-squares values of alpha and beta""" 
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta
least_squares_fit(x,y)


# In[69]:


b = estimate_coef(x,y)
b


# In[25]:


y_pred=df['temp'].apply(lambda a:b[0] + b[1]*a)
y_pred


# In[43]:


plt.scatter(x,y_pred,color= 'm')


# In[78]:


plt.xlabel('temperature')
plt.ylabel('area')
plt.plot(x,y_pred,color= 'm')
plt.show()


# In[79]:


## Using sklearn
from sklearn import linear_model


# In[111]:


lm = linear_model.LinearRegression()
model = lm.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))


# In[140]:


y_predicted = lm.predict(np.array(x).reshape(-1,1))
y_predicted


# In[144]:


plt.xlabel('temperature')
plt.ylabel('area')
plt.plot(x,y_predicted,color='m')
plt.show()


# In[ ]:




