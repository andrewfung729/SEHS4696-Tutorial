## T06 - Clustering using K-Means
#### The code is based on Chapter 10 Unsupervised Learning—Clustering Using K-Means in Lee (2019). 
#### Some lines have been modified to help students to understand better.

# You must follow the instructions that begin with "#----"

# For fill-in-the-blanks (i.e. _____) , use the words from the following line 
# bmxleg bmxwaist centroids column dataframe feature KMeans labels matrix Train

###1. Import the libraries and read the data file
import numpy as np
import pandas as pd

df = pd.read_csv("BMX_G.csv")

####2. Data Preprocessing
#---- Remove the print lines in your submission
print (df.shape)
print (df.isnull().sum()) # Find the number of NaNs in each ___ in the ___

df = df.dropna(subset=['bmxleg','bmxwaist']) #remove rows if there are NaNs in the columns bmxleg and bmxwaist

#---- Remove the print lines in your submission
print(df.shape)
print (df.isnull().sum()) #Check the number of NaNs in each column after removing NaNs


####3. Data Visualization Before Clustering
import matplotlib.pyplot as plt
plt.scatter(df['bmxleg'],df['bmxwaist'], c='r', s=2) #plot points using bmxleg as x-coordinate, and bmxwaist as y-coordinate
plt.xlabel("Upper leg Length (cm)")
plt.ylabel("Waist Circumference (cm)")


####4. Clustering using K-Means, set the value of K to 2 
from sklearn.cluster import KMeans
#---- Change the number of clusters to 4 in your submission
k = 2
X = df[['bmxleg','bmxwaist']].values #Create the features maxtrix using the values of the bmxleg and bmxwaist columns
m = KMeans(n_clusters=k)  #Create an object of the ____ class, and set its n_clusters attribute to k 
mTrained = m.fit(X)  #____ the new object using the data from the ____ ____
lbs  = mTrained.predict(X)  #Create a array of ____ which refers to which cluster a point in the dataset belong to
ctrs = mTrained.cluster_centers_ #Create a array of the ___.

####5. Data Visualization After Clustering
mycolors = ['b','r','y','g','c','m']
colors = [ mycolors[i] for i in lbs]
#---- Remove the print line in your submission
print (colors)

import matplotlib.pyplot as plt
plt.scatter(df['bmxleg'],df['bmxwaist'],c=colors,s=5)
plt.scatter(ctrs[:,0],ctrs[:,1], marker='*',s=200, c='y' )

#---- Print the coordinates of the centroids

#---- Fill in the blanks below using the column names in the df.
#The x-coordinate is the ______, and the y-coordinate is the ______


####6. Finding the Optimal K using Silhouette Scores as a metric
#---- Remove part 6 in your submission

from sklearn import metrics
qly_avgs_of_myClusters = []
min_k = 2
#try k from f2 to maximum number of labels
for k in range (min_k, 10):
    km = KMeans(n_clusters=k).fit(X)
    score = metrics.silhouette_score(X,km.labels_)
    print ("Silhouette coefficients for k = ", k, "is", score)
    qly_avgs_of_myClusters.append(score)

optimal_K = qly_avgs_of_myClusters.index(max(qly_avgs_of_myClusters)) + min_k
print ("The optimal value of K is ", optimal_K)

#The End
