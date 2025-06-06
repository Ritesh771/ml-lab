import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

np.random.seed(42)
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.1, size=x.shape)

def locally_weighted_regression(x_train, y_train, x_test, tau):
    m = len(x_train)
    y_pred = np.zeros_like(x_test)
    for i, x0 in enumerate(x_test):
        weights = np.exp(-((x_train - x0) ** 2) / (2 * tau ** 2))
        W = np.diag(weights)
        X_b = np.c_[np.ones((m, 1)), x_train] 
        theta = inv(X_b.T @ W @ X_b) @ (X_b.T @ W @ y_train)
        y_pred[i] = np.array([1, x0]) @ theta
    return y_pred

tau_values = [0.1, 0.5, 1, 5]
plt.figure(figsize=(10, 6))
for tau in tau_values:
    y_pred = locally_weighted_regression(x, y, x, tau)
    plt.plot(x, y_pred, label=f'tau={tau}')

plt.scatter(x, y, color='black', s=10, label='Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Locally Weighted Regression')
plt.legend()
plt.show()
