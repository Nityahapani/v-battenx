from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Callable

from .core import Booster


def train(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    evals: Optional[List[tuple]] = None,
    verbose_eval: int = 0,
    callbacks: Optional[List[Callable]] = None,
) -> Booster:
    p = dict(params or {})
    p.setdefault("verbose", verbose_eval)
    b = Booster(p)
    b.set_data(X, y)
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

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        b = train(X_tr, y_tr, params, num_boost_round)

        p_tr  = b.predict(X_tr)
        p_val = b.predict(X_val)

        def rmse(a, b_):
            return float(np.sqrt(np.mean((a - b_) ** 2)))

        train_scores.append(rmse(p_tr,  y_tr))
        val_scores.append(  rmse(p_val, y_val))

    return {"train-rmse-mean": train_scores, "test-rmse-mean": val_scores}
