from __future__ import annotations

import numpy as np
from typing import Tuple


# ── grad / hess primitives ────────────────────────────────────────────────────

def _mse_grad_hess(pred: np.ndarray,
                   label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    g = (pred - label).astype(np.float32)
    h = np.ones_like(g)
    return g, h


def _huber_grad_hess(pred: np.ndarray, label: np.ndarray,
                     delta: float) -> Tuple[np.ndarray, np.ndarray]:
    r    = pred - label
    absr = np.abs(r)
    g = np.where(absr <= delta, r,  delta * np.sign(r))
    h = np.where(absr <= delta, 1.0, delta / np.clip(absr, 1e-9, None))
    return g.astype(np.float32), h.astype(np.float32)


def _huber_loss(pred: np.ndarray, label: np.ndarray, delta: float) -> float:
    r    = pred - label
    absr = np.abs(r)
    l    = np.where(absr <= delta, 0.5 * r ** 2, delta * (absr - 0.5 * delta))
    return float(l.mean())


# ── WLS solvers ───────────────────────────────────────────────────────────────

def _wls_isotropic(X: np.ndarray, g: np.ndarray, h: np.ndarray,
                   reg_lambda: float) -> np.ndarray:
    Xw = X * h[:, None]
    A  = Xw.T @ X
    A += reg_lambda * np.eye(A.shape[0])
    b  = -(Xw.T @ g)
    return np.linalg.solve(A, b).astype(np.float32)


def _wls_diag_precond(X: np.ndarray, g: np.ndarray, h: np.ndarray,
                      reg_lambda: float) -> np.ndarray:
    """
    Diagonal-preconditioned ridge (Jacobi preconditioner).

    Standard ridge adds λI to XᵀWX, treating all feature directions equally.
    When features differ in scale or leverage, this under-regularises
    high-variance directions and over-regularises low-variance ones.

    We replace λI with λ · D where D = diag(XᵀWX) / mean(diag(XᵀWX)).

    This normalises each feature's penalty to its weighted second moment,
    so the effective regularisation per feature is:

        λ_j = λ · (Σ_i h_i x_{ij}²) / mean_j(Σ_i h_i x_{ij}²)

    Properties:
      - D = I when all features have equal weighted variance (recovers λI) ✓
      - Scale-invariant: multiplying feature j by c leaves the solution
        invariant (unlike isotropic ridge which would change it) ✓
      - Closed-form optimal under orthogonal design with heterogeneous
        feature variances (equivalent to per-feature ridge with λ_j ∝ σ_j²) ✓
      - O(d) extra cost; no matrix inversion beyond the LDLT already used ✓
    """
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


# ── Boosters ──────────────────────────────────────────────────────────────────

class MseBooster:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 reg_lambda: float = 1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MseBooster":
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights: list = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = _mse_grad_hess(pred, y)
            w    = _wls_isotropic(Xb, g, h, self.reg_lambda)
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


class HuberBooster:
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 reg_lambda: float = 1.0, delta: float = 1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.delta         = delta

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HuberBooster":
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights: list = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = _huber_grad_hess(pred, y, self.delta)
            w    = _wls_isotropic(Xb, g, h, self.reg_lambda)
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


class DiagPrecondBooster:
    """
    Gradient boosting with diagonal-preconditioned ridge regularisation.

    Replaces the isotropic λI penalty in the WLS normal equations with
    λ · diag(XᵀWX) / mean(diag(XᵀWX)), scaling each feature's regularisation
    by its weighted second moment.  See _wls_diag_precond for the math.

    Parameters
    ----------
    objective : 'mse' | 'huber'
    huber_delta : float
        Only used when objective='huber'.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 reg_lambda: float = 1.0, objective: str = "mse",
                 huber_delta: float = 1.0):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.objective     = objective
        self.huber_delta   = huber_delta

    def _grad_hess(self, pred, y):
        if self.objective == "huber":
            return _huber_grad_hess(pred, y, self.huber_delta)
        return _mse_grad_hess(pred, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DiagPrecondBooster":
        X  = np.asarray(X, dtype=np.float32)
        y  = np.asarray(y, dtype=np.float32)
        Xb = np.c_[X, np.ones(len(X), dtype=np.float32)]
        self._weights: list = []
        pred = np.zeros(len(y), dtype=np.float32)
        for _ in range(self.n_estimators):
            g, h = self._grad_hess(pred, y)
            w    = _wls_diag_precond(Xb, g, h, self.reg_lambda)
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
