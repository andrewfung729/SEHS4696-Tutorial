import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MEDV'] = housing.target
X = pd.DataFrame(np.c_[df['Population'], df['AveRooms']], columns=['Population', 'AveRooms'])
y = df['MEDV']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

myFeatures = PolynomialFeatures(degree=2)
X_train_poly = myFeatures.fit_transform(X_train)
print(myFeatures.get_feature_names_out(['Population', 'AveRooms']))

m = LinearRegression()
m.fit(X_train_poly, y_train)

X_test_poly = myFeatures.fit_transform(X_test)
print('R-Squared: %.4f' % m.score(X_test_poly, y_test))
