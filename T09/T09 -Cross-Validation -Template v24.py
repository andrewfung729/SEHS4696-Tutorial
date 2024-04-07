"""
Parameter-Tuning K in KNN using cross-validation
"""
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

iris = datasets.load_iris()

#---empty list that will hold cv (cross-validates) scores---
cv_scores = []

#---use all features---
X = iris.data[:, :4]
y = iris.target

#---number of folds---
folds = 10

#---creating odd list of K for KNN---
ks = list(range(1,int(len(X) * ((folds - 1)/folds))))

#---remove all multiples of 3---
# --vvv Enter your code below and execute the cell


#---perform k-fold cross validation---
for k in ks:
  knn = KNeighborsClassifier(n_neighbors=k)
  #---performs cross-validation and returns the average accuracy---
  #   NOTE: You can replace "accuracy" with other metrics such as "recall_macro"
  scores = cross_val_score(knn, X, y, cv=folds, scoring='accuracy')
  mean = scores.mean()
  cv_scores.append(mean)

# ## Finding the Optimal K


#---calculate misclassification error for each k---
MSE = [1 - x for x in cv_scores]
print (MSE)


#---calculate misclassification error for each k---
MSE = [1 - x for x in cv_scores]
#---determining best k (min. MSE)---
optimal_k = ks[MSE.index(min(MSE))]
print(f"The optimal number of neighbors is {optimal_k}")
#---plot misclassification error vs k---
plt.plot(ks, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


# # The End
