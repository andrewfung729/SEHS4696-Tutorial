from sklearn.datasets import load_breast_cancer 
from sklearn import linear_model
from sklearn.model_selection import train_test_split

ds = load_breast_cancer()
train_set, test_set, train_labels, test_labels = train_test_split(
            ds.data,    		  # features
			ds.target, 			  # labels
			test_size = 0.25,	  # split ratio
			random_state = 1, 	  # set random seed
			stratify = ds.target) # keep the same ratios 

x = train_set[:, 0:30]
y = train_labels
m = linear_model.LogisticRegression()
m.fit(x, y)
print (m.predict_proba(test_set))
print (m.predict(test_set))


from sklearn import metrics
print (metrics.confusion_matrix(test_labels, m.predict(test_set)))