from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any

from .core    import Booster
from .training import train


def train_dask(
    X,
    y,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    num_workers: int = 4,
    verbose_eval: int = 0,
) -> Booster:
    try:
        import dask.array as da
        import dask
    except ImportError:
        raise ImportError("dask[distributed] is required: pip install dask[distributed]")

    X_np = np.asarray(da.compute(X)[0] if hasattr(X, "compute") else X,
                       dtype=np.float32)
    y_np = np.asarray(da.compute(y)[0] if hasattr(y, "compute") else y,
                       dtype=np.float32)

    n     = len(X_np)
    chunk = max(1, n // num_workers)
    chunks = []

    p = dict(params or {})
    p.setdefault("verbose", verbose_eval)

    @dask.delayed
    def train_shard(X_shard, y_shard):
        b = Booster(p)
        b.set_data(X_shard, y_shard)
        b.train(num_boost_round)
        return b.train_loss

    delayed_results = []
    for i in range(num_workers):
        start = i * chunk
        end   = n if i == num_workers - 1 else start + chunk
        delayed_results.append(train_shard(X_np[start:end], y_np[start:end]))

    dask.compute(*delayed_results)

    # Final model trained on full data
    return train(X_np, y_np, p, num_boost_round)


class DaskBooster:
    def __init__(self, params: Optional[Dict[str, Any]] = None,
                 num_boost_round: int = 100, num_workers: int = 4):
        self.params          = params or {}
        self.num_boost_round = num_boost_round
        self.num_workers     = num_workers
        self._booster: Optional[Booster] = None

    def fit(self, X, y) -> "DaskBooster":
        self._booster = train_dask(X, y, self.params,
                                    self.num_boost_round, self.num_workers)
        return self

    def predict(self, X) -> np.ndarray:
        if self._booster is None:
            raise RuntimeError("Call fit() before predict()")
        X_np = np.asarray(X, dtype=np.float32)
        return self._booster.predict(X_np)

    @property
    def booster(self) -> Optional[Booster]:
        return self._booster
