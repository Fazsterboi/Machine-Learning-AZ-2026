"""
Machine Learning A-Z
Logistic Regression

Author: Sayed Faraaz
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample training data
X = np.array([
    [20],
    [25],
    [30],
    [35],
    [40],
    [45]
])

y = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

prediction = model.predict([[33]])

print("Prediction:", prediction[0])
