#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


columns_names = ['user_id','item_id','rating','timestamp']


# In[4]:


df = pd.read_csv('u.data',sep='\t',names=columns_names)


# In[5]:


df.head()


# In[6]:


movie_titles = pd.read_csv('Movie_Id_Titles')


# In[7]:


movie_titles.head()


# In[8]:


df = pd.merge(df,movie_titles,on='item_id')


# In[9]:


df.head()


# In[10]:


sns.set_style('white')


# In[11]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[12]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[13]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[14]:


ratings.head()


# In[15]:


ratings['numofRatings'] = df.groupby('title')['rating'].count()


# In[16]:


ratings.head()


# In[17]:


ratings['numofRatings'].hist(bins=70)


# In[18]:


ratings['rating'].hist(bins=70)


# In[19]:


sns.jointplot(x='rating',y='numofRatings',data=ratings,alpha=0.5)


# In[20]:


moviemat = df.pivot_table(index='user_id',columns='title',values='rating')


# In[21]:


moviemat.head()


# In[22]:


ratings.sort_values('numofRatings',ascending=False).head(10)


# In[23]:


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[24]:


starwars_user_ratings.head()


# In[25]:


similar_to_star_wars = moviemat.corrwith(starwars_user_ratings)


# In[26]:


similar_to_liar_liar = moviemat.corrwith(liarliar_user_ratings)


# In[28]:


corr_starwars = pd.DataFrame(similar_to_star_wars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)


# In[29]:


corr_starwars.head()


# In[30]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[32]:


corr_starwars = corr_starwars.join(ratings['numofRatings'])


# In[33]:


corr_starwars.head()


# In[34]:


corr_starwars[corr_starwars['numofRatings']>100].sort_values('Correlation',ascending=False).head()


# In[35]:


corr_liarliar = pd.DataFrame(similar_to_liar_liar,columns=['Correlation'])


# In[37]:


corr_liarliar.dropna(inplace=True)


# In[38]:


corr_liarliar = corr_liarliar.join(ratings['numofRatings'])


# In[41]:


corr_liarliar[corr_liarliar['numofRatings']>100].sort_values('Correlation',ascending=False).head()


# In[ ]:




