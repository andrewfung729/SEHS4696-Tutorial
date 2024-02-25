#!/usr/bin/env python
# coding: utf-8

# ## Multiple Linear Regression Exercise
#    
# #### Enter your name below as shown in the class register.
# #### Name: *CHAN Tai Man John*
# 
# ### In this exercise, you can replace the aliases/abbreviations and variable names as you see fit. You should add appropriate comments to help you and the reader to understand the program.
# References:<br>
# <ul>
#     <li>Ch6 of Lee (2019)</li>
#     <li>Agarwal (2018) at https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155</li>
# </ul>

# In[1]:


# You must download the dataset from the course web page.

# Import the necessary packages and give them aliases
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create a dataframe variable called df by calling the read_csv() function of the pandas package
df = pd.read_csv('California Housing.csv')

print (df.shape)

df.head()


# In[2]:


df.info()


# In[3]:


print (df.max())

print ()

df_new = df.isnull()
print (df_new)


# In[4]:


corr = df.corr()
print(corr)


# In[5]:


# Let's show the same dataframe in a hotter way - using a heatmap
import seaborn as sns
sns.heatmap(data=corr, annot=True)


# In[6]:


print(df.corr().abs().nlargest(3,'MEDV'))
print()
print(df.corr().abs().nlargest(3,'MEDV').index)
print()
print(df.corr().abs().nlargest(3,'MEDV').values[:,8])


# In[7]:


plt.title('Median income plotted against Median Value')
plt.xlabel('MedInc')
plt.ylabel('MEDV')
plt.plot(df['MedInc'],df['MEDV'], 'o')
plt.grid(True)


# In[8]:


plt.plot(df['AveRooms'],df['MEDV'], 'x')
plt.grid(True)


# In[9]:


X = pd.DataFrame(np.c_[df['AveRooms'],df['MedInc']],columns=['AveRooms','MedInc'])
print (X)
print ()

y = df['MEDV']
print (y)


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(X,y, test_size=0.2, random_state = 5 )


# In[11]:


slr_MedInc = LinearRegression()
slr_MedInc.fit(X=x_train[['MedInc']], y=Y_train)
# Try calculating the score 
print('R-Squared: %.4f' % slr_MedInc.score(x_test[['MedInc']],Y_test))

slr_rooms = LinearRegression()
slr_rooms.fit(X=x_train[['AveRooms']], y=Y_train)
print('R-Squared: %.4f' % slr_rooms.score(x_test[['AveRooms']],Y_test))


# # The End

# In[ ]:




