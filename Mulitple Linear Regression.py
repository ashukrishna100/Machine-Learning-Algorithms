#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[13]:


df = pd.read_csv("forestfires.csv")
df.head()


# Let us take FFMC,DMC,DC,ISI,temp,RH,wind,rain,area as X axis or dependent variables 
# 
# And area as Y axis or independent variable

# In[14]:


df_new = df.loc[:,'FFMC':'rain']
df_new.head()


# In[17]:


## y = b_0 + b_1x
x = df_new
y = df.area


# In[26]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)


# In[33]:


lm = linear_model.LinearRegression()
model = lm.fit(x,y)


# In[34]:


predictions = lm.predict(x_test)


# In[36]:


predictions[:20]


# In[39]:


# visualize
plt.scatter(y_test,predictions)


# ## check performance

# In[41]:


from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y_test,predictions)


# In[43]:


r2_score(np.array(y_test),predictions)


# In[ ]:




