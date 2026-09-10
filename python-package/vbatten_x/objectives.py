from __future__ import annotations

import numpy as np
from typing import Tuple


def _huber_grad_hess(pred: np.ndarray, label: np.ndarray,
                     delta: float) -> Tuple[np.ndarray, np.ndarray]:
    r    = pred - label
    absr = np.abs(r)
    g = np.where(absr <= delta, r,  delta * np.sign(r))
    h = np.where(absr <= delta, 1.0, delta / np.clip(absr, 1e-9, None))
    return g.astype(np.float32), h.astype(np.float32)


def _mse_grad_hess(pred: np.ndarray,
                   label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = (pred - label).astype(np.float32)
    h = np.ones_like(g)
    return g, h


def _huber_loss(pred: np.ndarray, label: np.ndarray, delta: float) -> float:
    r    = pred - label
    absr = np.abs(r)
    l    = np.where(absr <= delta, 0.5 * r ** 2, delta * (absr - 0.5 * delta))
    return float(l.mean())


def _wls_step(X: np.ndarray, g: np.ndarray, h: np.ndarray,
              reg_lambda: float) -> np.ndarray:
    Xw = X * h[:, None]
    A  = Xw.T @ X
    A += reg_lambda * np.eye(A.shape[0])
    b  = -(Xw.T @ g)
    return np.linalg.solve(A, b).astype(np.float32)


class HuberBooster:
    """
    Gradient boosting with Huber objective and sample-wise Hessian weighting.

    Uses the same WLS closed-form as LinearBooster::DoBoost — the key
    difference from the MSE objective is that h[i] = delta/|r_i| for
    outlier samples (|r_i| > delta), which down-weights their influence
    in the Newton step.

    Parameters
    ----------
    n_estimators : int
    learning_rate : float
    reg_lambda : float
    delta : float
        Huber threshold.  Residuals with |r| <= delta are treated
        quadratically; larger residuals are treated linearly.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 reg_lambda: float = 1.0, delta: float = 1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.delta         = delta
        self._weights: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HuberBooster":
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]

        self._weights = []
        pred = np.zeros(len(y), dtype=np.float32)

        for _ in range(self.n_estimators):
            g, h = _huber_grad_hess(pred, y, self.delta)
            w    = _wls_step(Xb, g, h, self.reg_lambda)
            pred = pred + self.learning_rate * (Xb @ w)
            self._weights.append(w)

        self._Xb_cols = Xb.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X  = np.asarray(X, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        out = np.zeros(len(X), dtype=np.float32)
        for w in self._weights:
            out += self.learning_rate * (Xb @ w)
        return out

    @property
    def train_loss(self) -> float:
        return self._last_loss if hasattr(self, "_last_loss") else float("nan")


class MseBooster:
    """Identical loop to HuberBooster but with MSE gradients (h=1 always)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 reg_lambda: float = 1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self._weights: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MseBooster":
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]

        self._weights = []
        pred = np.zeros(len(y), dtype=np.float32)

        for _ in range(self.n_estimators):
            g, h = _mse_grad_hess(pred, y)
            w    = _wls_step(Xb, g, h, self.reg_lambda)
            pred = pred + self.learning_rate * (Xb @ w)
            self._weights.append(w)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X  = np.asarray(X, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        out = np.zeros(len(X), dtype=np.float32)
        for w in self._weights:
            out += self.learning_rate * (Xb @ w)
        return out
