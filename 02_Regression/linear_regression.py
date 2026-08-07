# ============================================
# Linear Regression
# Machine Learning A-Z [2026]
# Author: Sayed Faraaz
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Train Model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[6]])

print("Prediction for x = 6:", prediction[0])

# Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()
