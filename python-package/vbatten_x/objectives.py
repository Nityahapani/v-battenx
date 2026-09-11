from __future__ import annotations

import numpy as np
from typing import Tuple


def _mse_grad_hess(pred, label):
    g = (pred - label).astype(np.float32)
    h = np.ones_like(g)
    return g, h


def _huber_grad_hess(pred, label, delta):
    r    = pred - label
    absr = np.abs(r)
    g = np.where(absr <= delta, r,  delta * np.sign(r))
    h = np.where(absr <= delta, 1.0, delta / np.clip(absr, 1e-9, None))
    return g.astype(np.float32), h.astype(np.float32)


def _huber_loss(pred, label, delta):
    r    = pred - label
    absr = np.abs(r)
    l    = np.where(absr <= delta, 0.5 * r ** 2, delta * (absr - 0.5 * delta))
    return float(l.mean())


def _mse_loss(pred, label):
    return float(np.mean((pred - label) ** 2) / 2)


def _wls_isotropic(X, g, h, reg_lambda):
    Xw = X * h[:, None]
    A  = Xw.T @ X
    A += reg_lambda * np.eye(A.shape[0])
    b  = -(Xw.T @ g)
    return np.linalg.solve(A, b).astype(np.float32)


def _wls_diag_precond(X, g, h, reg_lambda):
    Xw   = X * h[:, None]
    A    = Xw.T @ X
    diag = np.diag(A).copy()
    mean_diag = diag.mean()
    if mean_diag < 1e-9:
        A += reg_lambda * np.eye(A.shape[0])
    else:
        A += reg_lambda * np.diag(diag / mean_diag)
    b = -(Xw.T @ g)
    return np.linalg.solve(A, b).astype(np.float32)


def _wls_cholesky_free_intercept(X, g, h, reg_lambda):
    """
    WLS normal equations solved via Cholesky with an unregularised intercept.

    Two improvements over _wls_isotropic:

    1. Unregularised intercept
       Standard ridge adds λI to XᵀWX, penalising the bias term equally
       with all feature weights. This shrinks the intercept toward zero,
       introducing mean-prediction bias without reducing variance.

       We instead add the penalty only to the top-left d×d feature block:

           A = XᵀWX + diag(λ, ..., λ, 0)                    (1)

       Under Frisch-Waugh-Lovell this is equivalent to centring features
       by their h-weighted mean, fitting penalised WLS, then recovering
       the intercept as ȳ_w - X̄_w · w_feat.

    2. Cholesky solve
       A is SPD. np.linalg.solve uses LAPACK _gesv (LU): O(d³/3) flops.
       Cholesky costs O(d³/6) — 2× fewer — and is more numerically stable
       for SPD matrices (no pivoting required).
    """
    n, p = X.shape
    d    = p - 1

    Xw = X * h[:, None]
    A  = (Xw.T @ X).astype(np.float64)

    penalty       = np.zeros(p, dtype=np.float64)
    penalty[:d]   = reg_lambda
    A            += np.diag(penalty)

    b = -(Xw.T @ g).astype(np.float64)
    L = np.linalg.cholesky(A)
    w = np.linalg.solve(L.T, np.linalg.solve(L, b))
    return w.astype(np.float32)


class MseBooster:
    def __init__(self, n_estimators=100, learning_rate=0.1, reg_lambda=1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda

    def fit(self, X, y):
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = _mse_grad_hess(pred, y)
            w    = _wls_isotropic(Xb, g, h, self.reg_lambda)
            pred = pred + self.learning_rate * (Xb @ w)
            self._weights.append(w)
        return self

    def predict(self, X):
        X  = np.asarray(X, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        out = np.zeros(len(X), dtype=np.float32)
        for w in self._weights:
            out += self.learning_rate * (Xb @ w)
        return out


class HuberBooster:
    def __init__(self, n_estimators=100, learning_rate=0.1, reg_lambda=1.0, delta=1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.delta         = delta

    def fit(self, X, y):
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = _huber_grad_hess(pred, y, self.delta)
            w    = _wls_isotropic(Xb, g, h, self.reg_lambda)
            pred = pred + self.learning_rate * (Xb @ w)
            self._weights.append(w)
        return self

    def predict(self, X):
        X  = np.asarray(X, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        out = np.zeros(len(X), dtype=np.float32)
        for w in self._weights:
            out += self.learning_rate * (Xb @ w)
        return out


class DiagPrecondBooster:
    """Gradient boosting with diagonal-preconditioned ridge regularisation."""

    def __init__(self, n_estimators=100, learning_rate=0.1, reg_lambda=1.0,
                 objective="mse", huber_delta=1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.objective     = objective
        self.huber_delta   = huber_delta

    def _grad_hess(self, pred, y):
        if self.objective == "huber":
            return _huber_grad_hess(pred, y, self.huber_delta)
        return _mse_grad_hess(pred, y)

    def fit(self, X, y):
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = self._grad_hess(pred, y)
            w    = _wls_diag_precond(Xb, g, h, self.reg_lambda)
            pred = pred + self.learning_rate * (Xb @ w)
            self._weights.append(w)
        return self

    def predict(self, X):
        X  = np.asarray(X, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        out = np.zeros(len(X), dtype=np.float32)
        for w in self._weights:
            out += self.learning_rate * (Xb @ w)
        return out


class CholBooster:
    """
    Gradient boosting with Cholesky-solved WLS and unregularised intercept.

    See _wls_cholesky_free_intercept for the mathematical details.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, reg_lambda=1.0,
                 objective="mse", huber_delta=1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.objective     = objective
        self.huber_delta   = huber_delta

    def _grad_hess(self, pred, y):
        if self.objective == "huber":
            return _huber_grad_hess(pred, y, self.huber_delta)
        return _mse_grad_hess(pred, y)

    def fit(self, X, y):
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = self._grad_hess(pred, y)
            w    = _wls_cholesky_free_intercept(Xb, g, h, self.reg_lambda)
            pred = pred + self.learning_rate * (Xb @ w)
            self._weights.append(w)
        return self

    def predict(self, X):
        X  = np.asarray(X, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        out = np.zeros(len(X), dtype=np.float32)
        for w in self._weights:
            out += self.learning_rate * (Xb @ w)
        return out
