"""
Machine Learning A-Z
Multiple Linear Regression

Author: Sayed Faraaz
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "Experience": [1, 2, 3, 4, 5],
    "Education": [10, 12, 13, 15, 16],
    "Salary": [30000, 40000, 50000, 60000, 70000]
}

dataset = pd.DataFrame(data)

X = dataset[["Experience", "Education"]]
y = dataset["Salary"]

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6, 18]])

print("Predicted Salary:", prediction[0])
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
