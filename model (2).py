#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# In[5]:


dt = pd.read_csv('./Clean_Dataset.csv')
dt = dt.drop('Unnamed: 0', axis = 1)
dt.head()


# In[6]:


cat = ['airline','source_city', 'departure_time', 'stops','arrival_time', 'destination_city', 'class']
cont  = ['duration', 'days_left']


# In[7]:


dt.isnull().sum()


# In[8]:


dt.shape


# In[10]:


for i in cat:
    print(dt[i].unique())


# In[11]:


#Making a copy of original dataset.
data_copy = dt.copy()


# In[15]:


mapping = {'one': 1, 'zero': 0, 'two_or_more': 2}
replace_func = np.vectorize(lambda x: mapping.get(x, -1))

dt['stops'] = replace_func(dt['stops'])
dt['stops']


# In[16]:


le = LabelEncoder()

for i in cat:
    dt[i] = le.fit_transform(dt[i])


# In[29]:


dt = dt.drop('flight', axis = 1)
x = dt.drop('price', axis = 1)
y = dt['price']


# In[30]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25, random_state=42)


# In[31]:


xtrain2 = sc.fit_transform(xtrain[cont])
xtest2 = sc.fit_transform(xtest[cont])


# In[34]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
print(r2_score(ypred,ytest))


# In[37]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 150 ,max_depth = 30, min_samples_leaf = 5, min_samples_split = 30)
rf.fit(xtrain, ytrain)
ypred = rf.predict(xtest)
print(r2_score(ypred,ytest))


# In[42]:


import pickle
pickle.dump(rf, open('model.pkl', 'wb'))

model0 = pickle.load(open('model.pkl','rb'))


# In[ ]:




