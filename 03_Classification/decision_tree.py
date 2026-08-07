"""
Machine Learning A-Z
Decision Tree

Author: Sayed Faraaz
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([
    [20],
    [25],
    [30],
    [35],
    [40],
    [45]
])

y = np.array([0,0,0,1,1,1])

model = DecisionTreeClassifier(random_state=42)
model.fit(X,y)

prediction = model.predict([[32]])

print("Prediction:", prediction[0])
