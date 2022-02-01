#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np


# In[11]:


data=pd.read_csv("skin.csv")


# In[12]:


data.shape


# In[13]:


data.info()


# In[14]:





# In[15]:


data.value_counts()


# In[16]:


data.describe()


# In[17]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[18]:


import numpy as np
def split_test_train(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


# In[19]:


train_set,test_set=split_test_train(data,0.1)
len(train_set)


# In[20]:


corr_matrix=data.corr()
corr_matrix


# In[21]:


from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(data,test_size=0.2,random_state=42)
len(train_set)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import numpy as np

digits=load_digits()


# In[23]:


x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.3)


# In[24]:


def get_score(model,x_train,x_test,y_train,y_test):
    model.fit(x_train,y_train)
    return model.score(x_test,y_test)


# In[25]:


get_score(LogisticRegression(),x_train,x_test,y_train,y_test)


# In[26]:


get_score(SVC(),x_train,x_test,y_train,y_test)


# In[27]:


get_score(RandomForestClassifier(),x_train,x_test,y_train,y_test)


# In[28]:


import matplotlib.pyplot as plt

plt.matshow(data.corr())
plt.show()


# In[29]:


from sklearn.model_selection import KFold
kf=KFold(n_splits=6)
kf


# In[30]:


for train_index,test_index in kf.split([1,2,3,4,5,6,7,8]):
    print(train_index,test_index)


# In[31]:


from sklearn.model_selection import cross_val_score


# In[32]:


cross_val_score(LogisticRegression(),digits.data,digits.target)


# In[33]:


from sklearn.model_selection import StratifiedShuffleSplit
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(data,data["prognosis"]):
    strat_train_set=data.loc[train_index]
    strat_test_set=data.loc[test_index]


# In[34]:


strat_test_set["prognosis"].value_counts()/len(strat_test_set)


# In[35]:


for set_ in (strat_train_set,strat_test_set):
    set_.drop("prognosis",axis=0,inplace=True)
    
print(set)


# In[36]:


from pandas.plotting import scatter_matrix


# In[ ]:





# In[ ]:


scatter_matrix(data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[ ]:




