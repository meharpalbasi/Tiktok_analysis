#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


tiktok = pd.read_csv('C:/Users/mehar/OneDrive/Documents/spotify/TikTok_songs_2022.csv')


# In[13]:


tiktok.head()


# In[14]:


#Check for null values
tiktok.isnull().sum()


# In[15]:


#how many rows and columns?
tiktok.info()


# In[16]:


#top 10 least popular songs:
sorted_tiktok = tiktok.sort_values('track_pop', ascending = True).head(10)
print(sorted_tiktok)


# In[18]:


#descriptive statistics 
tiktok.describe().transpose()


# In[25]:


#top 10 most popular tracks, in place does not change original data frame
most_popular = tiktok.query('track_pop>90', inplace=False).sort_values('track_pop', ascending =False)
print(most_popular[:10])


# In[26]:


tiktok[['artist_name']].iloc[18]


# In[33]:


#we need to convert the duration of the songs to seconds instead of milliseconds 
tiktok['duration']=tiktok['duration_ms'].apply(lambda x:round(x/1000))
tiktok.drop('duration_ms', inplace=True, axis=1)


# In[40]:


corr_tiktok = tiktok.drop(['mode', 'key'], axis=1).corr(method='pearson')
plt.figure(figsize=(14,6))
corr_tiktok.style.background_gradient(cmap='inferno').set_precision(2)


# In[42]:


#regression line between Loudness and Energy
plt.figure(figsize=(14,6))
sns.regplot(data=tiktok, y='loudness', x='energy', color='b').set(title='Loudness vs Energy Correlation')


# In[44]:


x = "danceability"
y = "valence"

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=False, figsize=(10, 5))
fig.suptitle("Histograms")
h = ax2.hist2d(tiktok[x], tiktok[y], bins=20)
ax1.hist(tiktok["energy"])

ax2.set_xlabel(x)
ax2.set_ylabel(y)

ax1.set_xlabel("energy")

plt.colorbar(h[3], ax=ax2)

plt.show()


# In[ ]:




