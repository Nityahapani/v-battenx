from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Callable

from .core    import Booster, PhysicalDataset
from .physics import PhysicsSpec


def train(
    X,
    y: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    verbose_eval: int = 0,
    callbacks: Optional[List[Callable]] = None,
    evals: Optional[List] = None,
) -> Booster:
    p = dict(params or {})
    p.setdefault("verbose", verbose_eval)

    if isinstance(X, PhysicalDataset):
        ds = X
        X_arr, y_arr = ds.X, ds.y
    else:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32) if y is not None else np.zeros(len(X_arr), dtype=np.float32)

    b = Booster(p)
    b.set_data(X_arr, y_arr)
    b.train(num_boost_round)
    return b


def train_with_physics(
    X,
    y: Optional[np.ndarray] = None,
    physics_spec: Optional[PhysicsSpec] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    verbose_eval: int = 0,
) -> Booster:
    p = dict(params or {})
    p.setdefault("verbose", verbose_eval)

    if isinstance(X, PhysicalDataset):
        X_arr, y_arr = X.X, X.y if X.y is not None else np.zeros(len(X.X), dtype=np.float32)
    else:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32) if y is not None else np.zeros(len(X_arr), dtype=np.float32)

    b = Booster(p)
    b.set_data(X_arr, y_arr)
    if physics_spec is not None:
        b.set_physics(physics_spec)
    b.train(num_boost_round)
    return b


def early_stopping(rounds: int, metric: str = "train_loss",
                   min_delta: float = 1e-6) -> "EarlyStopping":
    from .callback import EarlyStopping
    return EarlyStopping(rounds, min_delta)


def cv(
    X,
    y: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    nfold: int = 5,
    num_boost_round: int = 100,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, List[float]]:
    from sklearn.model_selection import KFold

    if isinstance(X, PhysicalDataset):
        X_arr = X.X
        y_arr = X.y if X.y is not None else np.zeros(len(X.X), dtype=np.float32)
    else:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

    kf = KFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
    train_scores: List[float] = []
    val_scores:   List[float] = []

    for tr, val in kf.split(X_arr):
        b = train(X_arr[tr], y_arr[tr], params, num_boost_round)
        def rmse(a, b_):
            return float(np.sqrt(np.mean((a - b_) ** 2)))
        train_scores.append(rmse(b.predict(X_arr[tr]),  y_arr[tr]))
        val_scores.append(  rmse(b.predict(X_arr[val]), y_arr[val]))

    return {"train-rmse-mean": train_scores, "test-rmse-mean": val_scores}
