#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[5]:


df = pd.read_csv("forestfires.csv")
df = df[['ISI','area']]
df.head()


# In[6]:


x = df.ISI
y = df.area


# In[7]:


##visualize the data
plt.scatter(x,y,color='m')
plt.ylim(-1, 10)
plt.show()


# In[32]:





# In[ ]:





# In[ ]:





# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# In[72]:


polynomial_features= PolynomialFeatures(degree=7)
x_poly = polynomial_features.fit_transform(np.array(x).reshape(-1,1))


# In[73]:


model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)


# In[74]:


y_poly_pred[:20]


# In[75]:


math.sqrt(mean_squared_error(y,y_poly_pred))


# In[76]:


r2_score(y,y_poly_pred)


# In[49]:


##visualize(degree = 2)
plt.scatter(x, y)
plt.plot(x, y_poly_pred, color='m')
plt.show()


# In[77]:


##visualize(degree = 1)
plt.scatter(x, y)
plt.plot(x, y_poly_pred, color='m')
plt.show()


# In[ ]:




