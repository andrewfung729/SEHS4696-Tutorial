 # from Lee (2019) Ch 7.
# p.161
from sklearn.datasets import load_breast_cancer
ds = load_breast_cancer() # Load dataset
x = ds.data[:,0]          # mean radius
y = ds.target             # 0: malignant, 1: benign

# p.162
from sklearn import linear_model
m = linear_model.LogisticRegression()
#---train the model---
m.fit( X = x.reshape(len(x),1), y = y)

# p.163
# Predict the probability of a tumour sample 
# being cancerous if the mean radius is 20 
print ( m.predict_proba([[20]]) ) 

