"""
Machine Learning A-Z
Polynomial Regression

Author: Sayed Faraaz
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 5, 10, 17, 26])

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

prediction = model.predict(poly.transform([[6]]))

print("Prediction:", prediction[0])
