from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from .core    import Booster, PhysicalDataset
from .physics import PhysicsSpec


def train(
    X,
    y: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 100,
    verbose_eval: int = 0,
) -> Booster:
    p = {**(params or {}), "verbose": verbose_eval}
    b = Booster(p)
    b.set_data(X, y)
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
    p = {**(params or {}), "verbose": verbose_eval}
    b = Booster(p)
    b.set_data(X, y)
    if physics_spec is not None:
        b.set_physics(physics_spec)
    b.train(num_boost_round)
    return b


def early_stopping(rounds: int, min_delta: float = 1e-6) -> "EarlyStopping":
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
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    K-fold cross-validation.

    Returns a dict with keys "<metric>-train-mean", "<metric>-train-std",
    "<metric>-val-mean", "<metric>-val-std" for each requested metric,
    plus raw per-fold lists under "<metric>-train" and "<metric>-val".

    Default metric is RMSE.
    """
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        raise ImportError("scikit-learn is required for cv(): pip install scikit-learn")

    metrics = metrics or ["rmse"]

    if isinstance(X, PhysicalDataset):
        X_arr = X.X
        y_arr = X.y if X.y is not None else np.zeros(len(X), dtype=np.float32)
    else:
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

    kf = KFold(n_splits=nfold, shuffle=shuffle, random_state=seed)

    fold_scores: Dict[str, Dict[str, List[float]]] = {
        m: {"train": [], "val": []} for m in metrics
    }

    for tr_idx, val_idx in kf.split(X_arr):
        booster = train(X_arr[tr_idx], y_arr[tr_idx], params, num_boost_round)
        tr_pred  = booster.predict(X_arr[tr_idx])
        val_pred = booster.predict(X_arr[val_idx])

        for metric in metrics:
            tr_s  = _compute_metric(metric, tr_pred,  y_arr[tr_idx])
            val_s = _compute_metric(metric, val_pred, y_arr[val_idx])
            fold_scores[metric]["train"].append(tr_s)
            fold_scores[metric]["val"].append(val_s)

    out: Dict[str, Any] = {}
    for metric, splits in fold_scores.items():
        for split, vals in splits.items():
            arr = np.array(vals, dtype=np.float64)
            out[f"{metric}-{split}"]      = list(arr)
            out[f"{metric}-{split}-mean"] = float(arr.mean())
            out[f"{metric}-{split}-std"]  = float(arr.std())
            legacy_split = "test" if split == "val" else split
            out[f"{legacy_split}-{metric}-mean"] = list(arr)
            out[f"{legacy_split}-{metric}-std"]  = list(np.abs(arr - arr.mean()))

    return out


def _compute_metric(name: str, pred: np.ndarray, label: np.ndarray) -> float:
    pred  = pred.astype(np.float64)
    label = label.astype(np.float64)
    if name == "rmse":
        return float(np.sqrt(np.mean((pred - label) ** 2)))
    if name == "mae":
        return float(np.mean(np.abs(pred - label)))
    if name == "r2":
        ss_res = np.sum((label - pred) ** 2)
        ss_tot = np.sum((label - label.mean()) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-12))
    raise ValueError(f"Unknown metric: {name!r}. Supported: rmse, mae, r2")
