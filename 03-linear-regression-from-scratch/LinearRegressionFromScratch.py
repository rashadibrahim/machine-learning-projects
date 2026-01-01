import numpy as np


def linear_regressor(
    x: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    iterations: int = 10000,
    tol: float = 1e-3,
):

  x = np.asarray(x, dtype=float)
  y = np.asarray(y, dtype=float)
  if x.ndim == 1:
    x = x.reshape(-1, 1)
  if y.ndim == 1:
    y = y.reshape(-1, 1)
  if y.shape[0] != x.shape[0]:
    raise ValueError("x and y must have the same number of rows")

  samples_count, features_count = x.shape
  w = np.zeros((features_count, 1), dtype=float)
  b = 0.0

  for step in range(iterations):
    pred = np.dot(x, w) + b
    error = pred - y

    grad_w = (x.T @ error) / samples_count
    grad_b = np.sum(error) / samples_count

    w -= lr * grad_w
    b -= lr * grad_b

    if np.abs(error).mean() < tol:
      print(f"Converged after {step + 1} steps.")
      break

  return w, b



def predict(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
  if x.ndim == 1:
    x = x.reshape(-1, 1)
  if x.shape[1] != w.shape[0]:
    raise ValueError("x and w have incompatible shapes")
  x = np.asarray(x, dtype=float)
  return np.dot(x, w) + b

if __name__ == "__main__":

  x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
  y = np.array([3, 5, 7, 9, 11, 13], dtype=float)

  w, b = linear_regressor(x, y)
  print(f"Weights:\n{w}")
  print(f"Bias: {b:.4f}")
  new_x = np.array([7, 8, 9], dtype=float)
  print(f"Predictions: {predict(new_x, w, b).flatten()}")
  print('*' * 20)

  # Example with multiple features
  from sklearn.datasets import load_iris
  data = load_iris()
  x = data.data
  y = data.target
  w, b = linear_regressor(x, y, lr=0.01, iterations=5000)
  print(f"Weights:\n{w}")
  print(f"Bias: {b:.4f}")
  new_x = x[:5]
  print(f"Predictions: {predict(new_x, w, b).flatten()}")
  print(f"Actual: {y[:5]}")








