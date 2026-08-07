"""
Machine Learning A-Z
Artificial Neural Network (ANN)

Author: Sayed Faraaz
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])

model = Sequential()

model.add(Dense(4, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X, y, epochs=100, verbose=0)

prediction = model.predict(X)

print(prediction)
