
# coding: utf-8

# In[10]:


import pandas as pd
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# In[11]:


data = pd.read_table("hash_top.tsv", header = None)


# In[12]:


data.head()


# In[13]:


data[1].hist(bins = 20)    


# In[ ]:


ss = []
for k in range(2,11):
    modle = KMeans(n_clusters = k)
    y = modle.fit_predict(data[1].values[:,None])
    cluster_centers = modle.cluster_centers_
    print (1)
    C = np.apply_along_axis(lambda x:cluster_centers[x][0], 0, y)
    print(11)
    A = np.sum((C - data[1].values)**2)
    ss.append(A)
    
    


# In[17]:


plt.plot(range(2,11),ss)


# In[18]:


modle.cluster_centers_


# In[19]:


data.shape


# In[20]:


ss[0]/15858409


# In[21]:


data[1].max()


# In[31]:


ss = []
for k in range(2,11):
    modle = KMeans(n_clusters = k)
    y = modle.fit_predict(data[1].values[:,None])
    ss.append(modle.inertia_)
    print(k - 1)
    
    


# In[22]:


plt.plot(range(2,11),ss)


# In[23]:


data['cluster'] = y


# In[24]:


data.to_csv('M1')


# In[ ]:


with open ('clusters_centers','w') as file:
    for i, cluster_center in enumerate (modle.cluster_centers_):
        print (i,cluster_center[0],file = file)


# In[40]:


len(data[data['cluster']==0])


# In[38]:


len(data)

