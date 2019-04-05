#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("forestfires.csv")
df.head()


# In[4]:


df_new = df.loc[:,'FFMC':'rain']
df_new.head()


# In[5]:


## y = b_0 + b_1x
x = df_new
y = df.area


# In[13]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# In[7]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=101)


# In[8]:


lasso = linear_model.Lasso(alpha=0.0001, max_iter=10e5)
lasso.fit(x_train,y_train)


# In[9]:


print("training score",lasso.score(x_train,y_train))
print("test score",lasso.score(x_test,y_test))


# In[11]:


y_predict = lasso.predict(x_test)


# In[17]:


mean_squared_error(y_test,y_predict)


# In[18]:


r2_score(y_test,y_predict)


# ## Ridge

# In[19]:


from sklearn.linear_model import Ridge


# In[20]:


rr = Ridge(alpha=0.01)


# In[22]:


rr.fit(x_train, y_train)


# In[24]:


y_pred = rr.predict(x_test)


# In[25]:


mean_squared_error(y_test,y_pred)


# In[26]:


r2_score(y_test,y_pred)


# In[27]:


print("training score",rr.score(x_train,y_train))
print("test score",rr.score(x_test,y_test))


# In[ ]:




