
import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z))
    )


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    steps: int = 1000
) -> tuple[np.ndarray, float]:

    m, n = X.shape

    w = np.zeros(n)
    b = 0.0

    for _ in range(steps):

        z = X @ w + b
        y_pred = _sigmoid(z)

        error = y_pred - y

        dw = (X.T @ error) / m
        db = np.sum(error) / m

        w -= lr * dw
        b -= lr * db

    return w, b
