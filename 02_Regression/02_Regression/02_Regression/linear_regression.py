"""
Machine Learning A-Z
Linear Regression Example

Author: Sayed Faraaz
"""

import numpy as np
from sklearn.linear_model import LinearRegression

# Sample training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 4, 6])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict value
x_new = np.array([[6]])
prediction = model.predict(x_new)

print("Prediction for x = 6:", prediction[0])

# Display model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
