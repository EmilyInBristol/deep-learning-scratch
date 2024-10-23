import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

X = np.random.rand(100, 1) * 10
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

gbdt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbdt.fit(X_train, y_train)

y_pred = gbdt.predict(X_test)
print(y_pred)
print(y_test)

mse = mean_squared_error(y_test, y_pred)
print(mse)

# Plot the results
plt.scatter(X_test, y_test, color='black', label='Data')
plt.scatter(X_test, y_pred, color='red', label='Predictions')
plt.title('GBDT Regression Example')
plt.xlabel('Input Feature')
plt.ylabel('Target Variable')
plt.legend()
plt.show()


