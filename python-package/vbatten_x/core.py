from __future__ import annotations

import json
import numpy as np
from typing import Optional, Dict, Any, List

from . import _libvbatten as _lib


class MutationEvent:
    def __init__(self, d: Dict[str, Any]):
        self.type       = d.get("type", "no_op")
        self.region     = d.get("region", 0)
        self.pde_r      = d.get("pde_r", 0.0)

    def __repr__(self) -> str:
        return f"MutationEvent(type={self.type}, region={self.region}, pde_r={self.pde_r:.4f})"


class Booster:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self._params = params or {}
        self._handle = _lib.create(json.dumps(self._params))
        self._last_model_json: Optional[str] = None

    def __del__(self):
        if hasattr(self, "_handle") and self._handle is not None:
            _lib.destroy(self._handle)
            self._handle = None

    def set_data(self, X: np.ndarray, y: np.ndarray) -> "Booster":
        _lib.set_data(self._handle,
                      np.asarray(X, dtype=np.float32, order="C"),
                      np.asarray(y, dtype=np.float32))
        return self

    def set_physics(self, spec) -> "Booster":
        from .physics import PhysicsSpec
        s = spec.to_json() if isinstance(spec, PhysicsSpec) else json.dumps(spec)
        _lib.set_physics(self._handle, s)
        return self

    def train(self, n_iters: int = 100) -> "Booster":
        _lib.train(self._handle, n_iters)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _lib.predict(self._handle,
                            np.asarray(X, dtype=np.float32, order="C"))

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
        result = []
        for stage in data.get("stages", []):
            events = [MutationEvent(e) for e in stage.get("mutations", [])]
            result.append(events)
        return result

    def get_stage_pde_residuals(self) -> List[Dict[str, float]]:
        if self._last_model_json is None:
            return []
        data = json.loads(self._last_model_json)
        return [
            {"before": s.get("pde_before", 0.0), "after": s.get("pde_after", 0.0)}
            for s in data.get("stages", [])
        ]

    @property
    def train_loss(self) -> float:
        return _lib.train_loss(self._handle)

    @property
    def num_stages(self) -> int:
        return _lib.num_stages(self._handle)

    def __repr__(self) -> str:
        return (f"Booster(stages={self.num_stages}, "
                f"loss={self.train_loss:.6f}, params={self._params})")
