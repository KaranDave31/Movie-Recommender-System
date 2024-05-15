#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


##list of column names

columns_names = ['user_id','item_id','rating','timestamp']


# In[3]:


##reading data from - u.data, '\t' is tab separated delimiters, use column_names list created above as column names for dataframe

df = pd.read_csv('u.data',sep='\t',names=columns_names)


# In[4]:


## used to display first 5 rows from the data frame

df.head()


# In[5]:


## read data from csv file

movie_titles = pd.read_csv('Movie_Id_Titles')


# In[6]:


## display first 5 rows from the data frame

movie_titles.head()


# In[7]:


##merge two data frames on common column name item id

df = pd.merge(df,movie_titles,on='item_id')


# In[8]:


## display first five rows of data frame

df.head()


# In[9]:


##sets style for seaborn as white

sns.set_style('white')


# In[10]:


## group df dataframe by title, then calculate mean rating for each title, then sort the values in descending order and display first 5 rows of the data frame 

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# In[11]:


## group df dataframe by title, then count number of ratings for each title,then sort the values in descending order and display first five rows of dataframe 

df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# In[12]:


## create new dataframe called ratings, which stores mean value for ratings of each movie title

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[13]:


##display first five rows of the data frame

ratings.head()


# In[14]:


## creating new column in ratings data frame, which has total counts of each rating for a particular movie

ratings['numofRatings'] = df.groupby('title')['rating'].count()


# In[15]:


##display first five rows of the data frame

ratings.head()


# In[16]:


## histogram of num of ratings from ratings, no of intervals is specified as 70

ratings['numofRatings'].hist(bins=70)


# In[17]:


##histogram for rating from ratings, having intervals 70

ratings['rating'].hist(bins=70)


# In[18]:


## joint plot having rating on x and no of ratings on y, alpha adjusts the transparency of data points

sns.jointplot(x='rating',y='numofRatings',data=ratings,alpha=0.5)


# In[20]:


## makes a pivot table called moviemat where rows represent user id and columns represent title, values represent rating. 
## This pivot table essentially reshapes the data so that each row corresponds to a user, each column corresponds to a movie, and each cell contains the rating that the user gave to that movie.

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')


# In[21]:


## displays first five rows from the dataframe

moviemat.head()


# In[22]:


## sorts ratings data frame by num of ratings in a descending order, displays first 5 rows of the dataframe

ratings.sort_values('numofRatings',ascending=False).head(10)


# In[23]:


## extracts star wars 1977 column from moviemat dataframe and stores it, basically represents all the ratings given by users to star wars 1977

starwars_user_ratings = moviemat['Star Wars (1977)']

## extracts liar liar column from moviemat dataframe and stores it, basically represents all the ratings given by users to liar liar

liarliar_user_ratings = moviemat['Liar Liar (1997)']


# In[24]:


## displays first five rows from the dataframe

starwars_user_ratings.head()


# In[25]:


##calculate correlation between star wars and all other films 

similar_to_star_wars = moviemat.corrwith(starwars_user_ratings)


# In[26]:


##calculate correlation between liar liar and all other films 

similar_to_liar_liar = moviemat.corrwith(liarliar_user_ratings)


# In[27]:


## creates a new dataframe containing correlations and drops all null values
 
corr_starwars = pd.DataFrame(similar_to_star_wars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)


# In[29]:


corr_starwars.head()


# In[28]:


## sorts corr stars on basis correlation in a descending order and displays first five rows from the dataframe

corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[29]:


## join corr star wars with num of ratings of ratings table 

corr_starwars = corr_starwars.join(ratings['numofRatings'])


# In[33]:


corr_starwars.head()


# In[30]:


## filters corr star wars to include rows where num of ratings is greater than 100, sort correlation column values in descending order and print first five rows of it 

corr_starwars[corr_starwars['numofRatings']>100].sort_values('Correlation',ascending=False).head()


# In[31]:


## create new data frame for correlation of liar liar and all other films

corr_liarliar = pd.DataFrame(similar_to_liar_liar,columns=['Correlation'])


# In[32]:


## drop all rows having null values

corr_liarliar.dropna(inplace=True)


# In[33]:


## join corr liar liar with ratings dataframe with column num of ratings

corr_liarliar = corr_liarliar.join(ratings['numofRatings'])


# In[41]:


## filters corr liar liar to include rows where num of ratings is greater than 100, sort correlation column values in descending order and print first five rows of it 

corr_liarliar[corr_liarliar['numofRatings']>100].sort_values('Correlation',ascending=False).head()


# In[34]:


df.head()


# In[35]:


## calculate number of unique users in dataframe and print them 

n_users = df.user_id.nunique()
n_items = df.item_id.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Movies: '+str(n_items))


# In[36]:


## import train test split and spilt data into 75% for training and 25% for testing 

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size=0.25)


# In[37]:


#Create two user-item matrices, one for training and another for testing

## numpy array of zeros having dimension n_users and n_items
train_data_matrix = np.zeros((n_users, n_items))

## iterates over train data, and store values from userid, itemid, rating into the matrix
## Since user and item IDs start from 1 but array indices start from 0, subtracting 1 from the user and item IDs to get the correct index in the matrix.

for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  

test_data_matrix = np.zeros((n_users, n_items))

for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


# In[38]:


from sklearn.metrics.pairwise import pairwise_distances

## calculates cosine similarity based on ratings in train data matrix
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

## does the same for this but uses transpose of the matrix instead
item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')


# In[39]:


def predict(ratings, similarity, type='user'):
   
    ## if type is equal to user, calculate mean rating for each user across all items
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        
        
        ## calculates difference between ratings and mean user ratings, and ensure that mean user rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        
        ##computes predicted ratings, calculates dot product between similarity and ratings diff then normalises result by dividing it by sum of absolute similarities for each user
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    
    elif type == 'item':
        ##computes the predictions using item-based collaborative filtering. It calculates the dot product between ratings and similarity, then normalizes the result by dividing it by the sum of absolute similarities for each item.
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    
    return pred


# In[40]:


''' generate predictions for the ratings of items (movies) using 
 item-based collaborative filtering. You're passing train_data_matrix as 
 the ratings matrix, item_similarity as the similarity matrix calculated 
 based on items, and specifying type='item' to indicate item-based 
 collaborative filtering
'''


item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user')


# In[41]:


from sklearn.metrics import mean_squared_error
from math import sqrt

'''
defines a function named rmse that takes two arguments: prediction 
and ground_truth. These arguments represent the predicted ratings 
and the actual ratings (ground truth) for comparison.
'''

def rmse(prediction, ground_truth):
    '''
 filtering out the predicted and ground truth ratings to include only 
 the non-zero elements. This is necessary because you typically only 
 want to consider ratings for items that were actually rated by users.
'''
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# In[42]:


print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# In[43]:


##calculating the sparsity level of the MovieLens100K dataset
'''
n_users * n_items gives the total number of possible ratings in 
the user-item matrix.

1.0 - len(df) / float(n_users * n_items) calculates the 
proportion of missing values.

round(..., 3) rounds the calculated sparsity value to 3 decimal places

'''


sparsity=round(1.0-len(df)/float(n_users*n_items),3)
print('The sparsity level of MovieLens100K is ' +  str(sparsity*100) + '%')


# In[44]:


import scipy.sparse as sp
from scipy.sparse.linalg import svds

#get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix=np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
print('User-based CF MSE: ' + str(rmse(X_pred, test_data_matrix)))


# In[ ]:




