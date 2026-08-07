"""
Machine Learning A-Z
K-Means Clustering

Author: Sayed Faraaz
"""

import numpy as np
from sklearn.cluster import KMeans

X = np.array([
    [1,2],
    [1,4],
    [1,0],
    [10,2],
    [10,4],
    [10,0]
])

model = KMeans(n_clusters=2, random_state=42)

model.fit(X)

print("Cluster Labels")

print(model.labels_)
