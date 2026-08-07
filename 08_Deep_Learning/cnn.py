"""
Machine Learning A-Z
Convolutional Neural Network (CNN)

Author: Sayed Faraaz
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

model = Sequential()

model.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        input_shape=(64,64,3)
    )
)

model.add(Flatten())

model.add(Dense(units=128, activation="relu"))

model.add(Dense(units=1, activation="sigmoid"))

print(model.summary())
