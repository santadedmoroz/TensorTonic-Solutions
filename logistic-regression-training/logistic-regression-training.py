import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    W = np.zeros(n_features)

    b = 0.0

    for i in range(steps):
        z = X @ W + b
        p = _sigmoid(z)

        dw = X.T @ (p - y) / n_samples
        db = np.mean(p - y)


        W-= lr * dw
        b -= lr * db
        
    return W, b
    