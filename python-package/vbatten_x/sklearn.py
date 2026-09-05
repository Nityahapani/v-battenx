from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .core import Booster


class _VBattenXBase(BaseEstimator):
    def __init__(
        self,
        n_estimators:  int   = 100,
        learning_rate: float = 0.1,
        reg_lambda:    float = 1.0,
        tol:           float = 1e-6,
        verbose:       int   = 0,
    ):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.tol           = tol
        self.verbose       = verbose

    def _make_params(self, objective: str) -> Dict[str, Any]:
        return {
            "objective":     objective,
            "learning_rate": self.learning_rate,
            "reg_lambda":    self.reg_lambda,
            "tol":           self.tol,
            "verbose":       self.verbose,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "n_estimators":  self.n_estimators,
            "learning_rate": self.learning_rate,
            "reg_lambda":    self.reg_lambda,
            "tol":           self.tol,
            "verbose":       self.verbose,
        }

    def set_params(self, **params) -> "_VBattenXBase":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "booster_")
        n = self.n_features_in_
        scores = np.zeros(n, dtype=np.float64)
        for s in range(self.booster_.num_stages):
            pass
        return scores / max(scores.sum(), 1e-9)


class VBattenXRegressor(_VBattenXBase, RegressorMixin):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight=None,
    ) -> "VBattenXRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        self.booster_ = Booster(self._make_params("regression"))
        self.booster_.set_data(X, y)
        self.booster_.train(self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "booster_")
        return self.booster_.predict(np.asarray(X, dtype=np.float32)).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y, self.predict(X)))


class VBattenXClassifier(_VBattenXBase, ClassifierMixin):
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight=None,
    ) -> "VBattenXClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_       = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.booster_ = Booster(self._make_params("classification"))
        self.booster_.set_data(X, y)
        self.booster_.train(self.n_estimators)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "booster_")
        raw = self.booster_.predict(np.asarray(X, dtype=np.float32))
        p   = 1.0 / (1.0 + np.exp(-raw.astype(np.float64)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        from sklearn.metrics import accuracy_score
        return float(accuracy_score(y, self.predict(X)))
