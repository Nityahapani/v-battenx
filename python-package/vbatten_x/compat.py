from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Optional


def to_numpy(pred) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64)


def from_pandas(df, label_col: Optional[str] = None):
    y = df[label_col].values.astype(np.float32) if label_col else None
    cols = [c for c in df.columns if c != label_col]
    return df[cols].values.astype(np.float32), y


def try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(f"{name} is required for this interop function")


def to_torch_tensor(arr: np.ndarray):
    torch = try_import("torch")
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def from_torch_tensor(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def to_jax_array(arr: np.ndarray):
    jnp = try_import("jax.numpy")
    return jnp.array(np.asarray(arr, dtype=np.float32))


def to_scipy_sparse(adjacency: dict):
    scipy_sparse = try_import("scipy.sparse")
    rows, cols, data = [], [], []
    for src, neighbours in adjacency.items():
        for dst, w in neighbours.items():
            rows.append(src); cols.append(dst); data.append(w)
    n = max(max(rows, default=0), max(cols, default=0)) + 1
    return scipy_sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def field_params_to_numpy(booster, stage: int = -1) -> np.ndarray:
    import json
    if booster._last_model_json is None:
        raise RuntimeError("Call booster.save() first")
    data   = json.loads(booster._last_model_json)
    stages = data.get("stages", [])
    s_idx  = stage if stage >= 0 else len(stages) + stage
    return np.array(stages[s_idx].get("params", []), dtype=np.float32)
