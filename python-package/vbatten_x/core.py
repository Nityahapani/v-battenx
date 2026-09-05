from __future__ import annotations

import json
import numpy as np
from typing import Optional, Dict, Any

from . import _libvbatten as _lib


class Booster:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params = params or {}
        self._handle = _lib.create(json.dumps(self._params))
        self._nrows  = 0
        self._ncols  = 0

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            _lib.destroy(self._handle)
            self._handle = None

    def set_data(self, X: np.ndarray, y: np.ndarray) -> "Booster":
        X = np.asarray(X, dtype=np.float32, order="C")
        y = np.asarray(y, dtype=np.float32)
        self._nrows, self._ncols = X.shape
        _lib.set_data(self._handle, X, y)
        return self

    def train(self, n_iters: int = 100) -> "Booster":
        _lib.train(self._handle, n_iters)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32, order="C")
        return _lib.predict(self._handle, X)

    def save(self, path: str) -> None:
        _lib.save(self._handle, path)

    @classmethod
    def load(cls, path: str, params: Optional[Dict[str, Any]] = None) -> "Booster":
        b = cls(params)
        _lib.load(b._handle, path)
        return b

    @property
    def train_loss(self) -> float:
        return _lib.train_loss(self._handle)

    @property
    def num_stages(self) -> int:
        return _lib.num_stages(self._handle)

    def __repr__(self) -> str:
        return (f"Booster(stages={self.num_stages}, "
                f"loss={self.train_loss:.6f}, params={self._params})")
