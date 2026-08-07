"""
Machine Learning A-Z
K-Nearest Neighbors (KNN)

Author: Sayed Faraaz
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

X = np.array([
    [20],
    [25],
    [30],
    [35],
    [40],
    [45]
])

y = np.array([0,0,0,1,1,1])

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

prediction = model.predict([[32]])

print("Prediction:", prediction[0])
