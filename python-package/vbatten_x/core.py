from __future__ import annotations

import json
import numpy as np
from typing import Optional, Dict, Any, List

from . import _libvbatten as _lib


class PhysicalDataset:
    def __init__(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        units: Optional[Dict[str, str]] = None,
    ):
        self.X             = np.asarray(X, dtype=np.float32, order="C")
        self.y             = np.asarray(y, dtype=np.float32) if y is not None else None
        self.feature_names = feature_names or [f"f{i}" for i in range(self.X.shape[1])]
        self.units         = units or {}

    @classmethod
    def from_numpy(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        units: Optional[Dict[str, str]] = None,
    ) -> "PhysicalDataset":
        return cls(X, y, feature_names, units)

    @classmethod
    def from_pandas(
        cls,
        df,
        label_col: Optional[str] = None,
        units: Optional[Dict[str, str]] = None,
    ) -> "PhysicalDataset":
        y    = df[label_col].values if label_col else None
        cols = [c for c in df.columns if c != label_col]
        return cls(df[cols].values, y, cols, units)

    @classmethod
    def from_csv(
        cls,
        path: str,
        label_col: Optional[str] = None,
        **kwargs,
    ) -> "PhysicalDataset":
        import pandas as pd
        return cls.from_pandas(pd.read_csv(path, **kwargs), label_col)

    @property
    def shape(self):
        return self.X.shape

    def __len__(self) -> int:
        return self.X.shape[0]

    def __repr__(self) -> str:
        suffix = "..." if len(self.feature_names) > 3 else ""
        return (
            f"PhysicalDataset(rows={self.X.shape[0]}, cols={self.X.shape[1]}, "
            f"features={self.feature_names[:3]}{suffix})"
        )


class MutationEvent:
    __slots__ = ("type", "region", "pde_r")

    def __init__(self, d: Dict[str, Any]):
        self.type   = d.get("type",   "no_op")
        self.region = d.get("region", 0)
        self.pde_r  = d.get("pde_r",  0.0)

    def __repr__(self) -> str:
        return f"MutationEvent(type={self.type}, region={self.region}, pde_r={self.pde_r:.4f})"


class Booster:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params = params or {}
        self._handle = _lib.create(json.dumps(self._params))
        self._last_model_json: Optional[str] = None

    def __del__(self):
        h = getattr(self, "_handle", None)
        if h is not None:
            _lib.destroy(h)
            self._handle = None

    def set_data(self, X, y: Optional[np.ndarray] = None) -> "Booster":
        if isinstance(X, PhysicalDataset):
            y_arr = X.y if X.y is not None else np.zeros(len(X), dtype=np.float32)
            _lib.set_data(self._handle, X.X, y_arr)
        else:
            X_arr = np.asarray(X, dtype=np.float32, order="C")
            y_arr = np.asarray(y, dtype=np.float32) if y is not None else np.zeros(len(X_arr), dtype=np.float32)
            _lib.set_data(self._handle, X_arr, y_arr)
        return self

    def set_physics(self, spec) -> "Booster":
        from .physics import PhysicsSpec
        s = spec.to_json() if isinstance(spec, PhysicsSpec) else json.dumps(spec)
        _lib.set_physics(self._handle, s)
        return self

    def train(self, n_iters: int = 100) -> "Booster":
        _lib.train(self._handle, n_iters)
        return self

    def predict(self, X) -> np.ndarray:
        arr = X.X if isinstance(X, PhysicalDataset) else np.asarray(X, dtype=np.float32, order="C")
        return _lib.predict(self._handle, arr)

    def save(self, path: str) -> None:
        _lib.save(self._handle, path)
        with open(path) as f:
            self._last_model_json = f.read()

    @classmethod
    def load(cls, path: str, params: Optional[Dict[str, Any]] = None) -> "Booster":
        b = cls(params)
        _lib.load(b._handle, path)
        with open(path) as f:
            b._last_model_json = f.read()
        return b

    def get_mutation_log(self) -> List[List[MutationEvent]]:
        if self._last_model_json is None:
            return []
        data = json.loads(self._last_model_json)
        return [
            [MutationEvent(e) for e in s.get("mutations", [])]
            for s in data.get("stages", [])
        ]

    def get_stage_pde_residuals(self) -> List[Dict[str, float]]:
        if self._last_model_json is None:
            return []
        data = json.loads(self._last_model_json)
        return [
            {"before": s.get("pde_before", 0.0), "after": s.get("pde_after", 0.0)}
            for s in data.get("stages", [])
        ]

    def get_metric(self, name: str) -> float:
        return _lib.get_metric(self._handle, name)

    @property
    def train_loss(self) -> float:
        return _lib.train_loss(self._handle)

    @property
    def num_stages(self) -> int:
        return _lib.num_stages(self._handle)

    @property
    def abi_version(self) -> int:
        return _lib.abi_version()

    @property
    def lib_version(self) -> str:
        return _lib.lib_version()

    def __repr__(self) -> str:
        return f"Booster(stages={self.num_stages}, loss={self.train_loss:.6f}, params={self._params})"
