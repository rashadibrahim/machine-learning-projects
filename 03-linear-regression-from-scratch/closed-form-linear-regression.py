import numpy as np

class LinearRegressionClosed:
  def __init__(self):
    self.coef_ = None
    self.intercept_ = 0.0

    
  def fit(self, X, y):
    X = np.array(X)
    y = np.array(y)
    
    Xb = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
    """
    Assuming Xb is the design matrix with a bias term and y is the target vector,
    the closed-form solution for linear regression coefficients A is given by:
    A = (Xb^T * Xb)^(-1) * Xb^T * y
    since RSS = ||y - XbA||^2, minimizing RSS 'RSS derivative == 0' leads to this formula.
    """
    A = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y

    self.intercept_ = A[0]
    self.coef_ = A[1:]

  def predict(self, X):
    X = np.array(X)
    return X @ self.coef_ + self.intercept_
  

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

reg = LinearRegressionClosed()
X, y = fetch_california_housing(return_X_y=True)
reg.fit(X, y)

y_pred = reg.predict(X)
print("R2 score (closed-form):", r2_score(y, y_pred))

sklearn_reg = LinearRegression()
sklearn_reg.fit(X, y)
y_sklearn_pred = sklearn_reg.predict(X)
print("R2 score (sklearn):", r2_score(y, y_sklearn_pred))