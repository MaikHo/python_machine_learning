from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import numpy as np

X = np.array([[0.44, 0.68], [0.99, 0.23]])
vector = np.array([[109.85], [155.72]])
predict= np.array([[0.49, 0.18]])

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X)
predict_ = poly.fit_transform(predict)

clf = linear_model.LinearRegression()
clf.fit(X_, vector)
print(clf.predict(predict_))
