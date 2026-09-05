from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Booster


def to_numpy(pred) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64)


def from_pandas(df, label_col: str | None = None):
    import pandas as pd
    y = df[label_col].values.astype(np.float32) if label_col else None
    cols = [c for c in df.columns if c != label_col]
    X = df[cols].values.astype(np.float32)
    return (X, y) if y is not None else X


def try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError("PyTorch is required for this interop function")


def to_torch_tensor(arr: np.ndarray):
    torch = try_import_torch()
    return torch.from_numpy(np.asarray(arr, dtype=np.float32))


def from_torch_tensor(t) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float32)


def to_scipy_sparse(adjacency: dict):
    try:
        from scipy.sparse import csr_matrix
    except ImportError:
        raise ImportError("scipy is required for this interop function")
    rows, cols, data = [], [], []
    for src, neighbours in adjacency.items():
        for dst, w in neighbours.items():
            rows.append(src); cols.append(dst); data.append(w)
    n = max(max(rows, default=0), max(cols, default=0)) + 1
    return csr_matrix((data, (rows, cols)), shape=(n, n))
