from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Callable

from .core    import Booster
from .physics import PhysicsSpec


def train(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    verbose_eval: int = 0,
    callbacks: Optional[List[Callable]] = None,
) -> Booster:
    p = dict(params or {})
    p.setdefault("verbose", verbose_eval)
    b = Booster(p)
    b.set_data(X, y)
    b.train(num_boost_round)
    return b


def train_with_physics(
    X: np.ndarray,
    y: np.ndarray,
    physics_spec: PhysicsSpec,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    verbose_eval: int = 0,
) -> Booster:
    p = dict(params or {})
    p.setdefault("verbose", verbose_eval)
    b = Booster(p)
    b.set_data(X, y)
    b.set_physics(physics_spec)
    b.train(num_boost_round)
    return b


def cv(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    nfold: int = 5,
    num_boost_round: int = 100,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, List[float]]:
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
    train_scores: List[float] = []
    val_scores:   List[float] = []

    for tr_idx, val_idx in kf.split(X):
        b = train(X[tr_idx], y[tr_idx], params, num_boost_round)
        def rmse(a, b_):
            return float(np.sqrt(np.mean((a - b_) ** 2)))
        train_scores.append(rmse(b.predict(X[tr_idx]), y[tr_idx]))
        val_scores.append(  rmse(b.predict(X[val_idx]), y[val_idx]))

    return {"train-rmse-mean": train_scores, "test-rmse-mean": val_scores}
