from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .core    import Booster
from .physics import PhysicsSpec


class _VBattenXBase(BaseEstimator):
    def __init__(
        self,
        n_estimators:  int   = 100,
        learning_rate: float = 0.1,
        reg_lambda:    float = 1.0,
        lambda_pde:    float = 0.0,
        tol:           float = 1e-6,
        verbose:       int   = 0,
        physics_spec:  Optional[PhysicsSpec] = None,
        dtdo:          str   = "none",
        tau_expand:    float = 0.1,
        tau_collapse:  float = 0.01,
        max_total_dim: int   = 64,
        max_regions:   int   = 16,
        ras_alpha:     float = 0.0,
        huber_delta:   float = 1.0,
        objective:     str   = "regression",
    ):
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda    = reg_lambda
        self.lambda_pde    = lambda_pde
        self.tol           = tol
        self.verbose       = verbose
        self.physics_spec  = physics_spec
        self.dtdo          = dtdo
        self.tau_expand    = tau_expand
        self.tau_collapse  = tau_collapse
        self.max_total_dim = max_total_dim
        self.max_regions   = max_regions
        self.ras_alpha     = ras_alpha
        self.huber_delta   = huber_delta
        self.objective     = objective

    def _make_params(self, objective: str) -> Dict[str, Any]:
        obj = self.objective if self.objective != "regression" else objective
        return {
            "objective":     obj,
            "learning_rate": self.learning_rate,
            "reg_lambda":    self.reg_lambda,
            "lambda_pde":    self.lambda_pde,
            "tol":           self.tol,
            "verbose":       self.verbose,
            "dtdo":          self.dtdo,
            "tau_expand":    self.tau_expand,
            "tau_collapse":  self.tau_collapse,
            "max_total_dim": self.max_total_dim,
            "max_regions":   self.max_regions,
            "ras_alpha":     self.ras_alpha,
            "huber_delta":   self.huber_delta,
        }

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in [
            "n_estimators", "learning_rate", "reg_lambda", "lambda_pde",
            "tol", "verbose", "physics_spec", "dtdo",
            "tau_expand", "tau_collapse", "max_total_dim", "max_regions",
            "ras_alpha", "huber_delta", "objective",
        ]}

    def set_params(self, **params) -> "_VBattenXBase":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        check_is_fitted(self, "booster_")
        return np.zeros(self.n_features_in_, dtype=np.float64)

    @property
    def mutation_log_(self):
        check_is_fitted(self, "booster_")
        return self.booster_.get_mutation_log()


class VBattenXRegressor(_VBattenXBase, RegressorMixin):
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight=None) -> "VBattenXRegressor":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        self.booster_ = Booster(self._make_params("regression"))
        self.booster_.set_data(X, y)
        if self.physics_spec is not None:
            self.booster_.set_physics(self.physics_spec)
        self.booster_.train(self.n_estimators)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "booster_")
        return self.booster_.predict(
            np.asarray(X, dtype=np.float32)).astype(np.float64)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y, self.predict(X)))


class VBattenXClassifier(_VBattenXBase, ClassifierMixin):
    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight=None) -> "VBattenXClassifier":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_       = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.booster_ = Booster(self._make_params("classification"))
        self.booster_.set_data(X, y)
        if self.physics_spec is not None:
            self.booster_.set_physics(self.physics_spec)
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
