#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[32]:


df = pd.read_csv("forestfires.csv")
df = df[['temp','area']]
df.head()


# In[33]:


## y = b_0 + b_1x
x = df.temp
y = df.area


# In[34]:


##visualize the data
plt.scatter(x,y,color='m')
plt.ylim(-1, 10)
plt.show()


# In[35]:


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


# In[36]:


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


# In[37]:


b = estimate_coef(x,y)
b


# In[38]:


y_pred=df['temp'].apply(lambda a:b[0] + b[1]*a)
y_pred


# In[39]:


plt.scatter(x,y_pred,color= 'm')


# In[40]:


plt.xlabel('temperature')
plt.ylabel('area')
plt.plot(x,y_pred,color= 'm')
plt.show()


# In[41]:


## Using sklearn
from sklearn import linear_model


# In[42]:


lm = linear_model.LinearRegression()
model = lm.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))


# In[43]:


y_predicted = lm.predict(np.array(x).reshape(-1,1))
y_predicted


# In[44]:


plt.xlabel('temperature')
plt.ylabel('area')
plt.plot(x,y_predicted,color='m')
plt.show()


# In[45]:


#coefficient/slope
print(lm.coef_)


# In[46]:


#intercept
print(lm.intercept_)


# ## Performance of the model

# In[47]:


## the higher the R-squared, the better the model fits your data
## r2 square score
from sklearn.metrics import mean_squared_error, r2_score
import math


# In[48]:


##lesser the RMSE better the model
mse = mean_squared_error(y , y_predicted)
print("mean squared error is :",mse)
print("root mean squared error is :",math.sqrt(mse))


# In[49]:


## higher the r2 score better the model
r2_score(y , y_predicted)


# ## GRADIENT DESCENT

# In[50]:


#### GRADIENT DESCENT ########

m = 0
c = 0

L = 0.001  # The learning Rate
epochs = 100  # The number of iterations to perform gradient descent

n = float(len(x)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*x + c  # The current predicted value of Y
    D_m = (-2/n) * sum(x * (y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)


# In[51]:


y_pred=df['temp'].apply(lambda a:c + m*a)
y_pred.head()


# In[52]:


plt.plot(x,y_pred,color='m')


# In[53]:


mean_squared_error(y , y_pred)


# In[ ]:





# In[ ]:




