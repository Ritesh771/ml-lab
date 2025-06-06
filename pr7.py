import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
X = california.data
y = california.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
print(f"Linear Regression MSE: {mse_linear:.2f}")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
auto_mpg = pd.read_csv(url, header=None, delim_whitespace=True, names=column_names)

auto_mpg['horsepower'] = pd.to_numeric(auto_mpg['horsepower'], errors='coerce')
auto_mpg = auto_mpg.dropna()

X_poly = auto_mpg[['weight']]
y_poly = auto_mpg['mpg']

X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y_poly, test_size=0.2, random_state=42)

poly = PolynomialFeatures(degree=3)
X_train_poly_transformed = poly.fit_transform(X_train_poly)
X_test_poly_transformed = poly.transform(X_test_poly)

polynomial_model = LinearRegression()
polynomial_model.fit(X_train_poly_transformed, y_train_poly)

y_pred_poly = polynomial_model.predict(X_test_poly_transformed)

mse_poly = mean_squared_error(y_test_poly, y_pred_poly)
print(f"Polynomial Regression MSE: {mse_poly:.2f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_test_poly, y_test_poly, color='blue', label='True values')
plt.scatter(X_test_poly, y_pred_poly, color='red', label='Predicted values')
plt.title("Polynomial Regression (Weight vs MPG)")
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_linear, color='green', label='True vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.title("Linear Regression (California Housing)")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()

plt.tight_layout()
plt.show()
