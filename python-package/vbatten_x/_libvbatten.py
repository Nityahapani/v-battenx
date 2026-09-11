from __future__ import annotations

import ctypes
import os
import sys
import numpy as np

_here = os.path.dirname(__file__)
_so   = os.path.join(_here, "_vbatten_x.so")

if not os.path.exists(_so):
    raise ImportError(f"Native library not found: {_so}")

_lib = ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)

_lib.vbx_learner_create.restype  = ctypes.c_void_p
_lib.vbx_learner_create.argtypes = [ctypes.c_char_p]

_lib.vbx_set_data.restype  = ctypes.c_int
_lib.vbx_set_data.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int64,
]

_lib.vbx_train.restype  = ctypes.c_int
_lib.vbx_train.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.vbx_predict.restype  = ctypes.c_int
_lib.vbx_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int64,
    ctypes.POINTER(ctypes.c_float),
]

_lib.vbx_save.restype  = ctypes.c_int
_lib.vbx_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.vbx_load.restype  = ctypes.c_int
_lib.vbx_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.vbx_set_physics.restype  = ctypes.c_int
_lib.vbx_set_physics.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.vbx_train_loss.restype  = ctypes.c_float
_lib.vbx_train_loss.argtypes = [ctypes.c_void_p]

_lib.vbx_num_stages.restype  = ctypes.c_int
_lib.vbx_num_stages.argtypes = [ctypes.c_void_p]

_lib.vbx_get_metric.restype  = ctypes.c_double
_lib.vbx_get_metric.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.vbx_get_mutation_log.restype  = ctypes.c_char_p
_lib.vbx_get_mutation_log.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.vbx_abi_version.restype  = ctypes.c_int
_lib.vbx_abi_version.argtypes = []

_lib.vbx_version_string.restype  = ctypes.c_char_p
_lib.vbx_version_string.argtypes = []

_lib.vbx_last_error.restype  = ctypes.c_char_p
_lib.vbx_last_error.argtypes = []

_lib.vbx_destroy.restype  = None
_lib.vbx_destroy.argtypes = [ctypes.c_void_p]


def _check(ret: int) -> None:
    if ret != 0:
        msg = _lib.vbx_last_error()
        raise RuntimeError(msg.decode() if msg else "vbx call failed")


def _f32c(arr: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _f32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32, order="C")


def create(params_json: str) -> ctypes.c_void_p:
    h = _lib.vbx_learner_create(params_json.encode())
    if h is None:
        msg = _lib.vbx_last_error()
        raise RuntimeError(msg.decode() if msg else "failed to create learner")
    return h


def set_data(h, X: np.ndarray, y: np.ndarray) -> None:
    X_ = _f32(X)
    y_ = _f32(y)
    nrows, ncols = X_.shape
    _check(_lib.vbx_set_data(h, _f32c(X_), _f32c(y_), nrows, ncols))


def train(h, n_iters: int) -> None:
    _check(_lib.vbx_train(h, int(n_iters)))


def predict(h, X: np.ndarray) -> np.ndarray:
    X_ = _f32(X)
    nrows, ncols = X_.shape
    out = np.empty(nrows, dtype=np.float32)
    _check(_lib.vbx_predict(h, _f32c(X_), nrows, ncols, _f32c(out)))
    return out


def set_physics(h, spec_json: str) -> None:
    _check(_lib.vbx_set_physics(h, spec_json.encode()))


def save(h, path: str) -> None:
    _check(_lib.vbx_save(h, path.encode()))


def load(h, path: str) -> None:
    _check(_lib.vbx_load(h, path.encode()))


def train_loss(h) -> float:
    return float(_lib.vbx_train_loss(h))


def num_stages(h) -> int:
    return int(_lib.vbx_num_stages(h))


def get_metric(h, name: str) -> float:
    return float(_lib.vbx_get_metric(h, name.encode()))


def get_mutation_log_json(h, stage: int) -> str:
    raw = _lib.vbx_get_mutation_log(h, int(stage))
    return raw.decode() if raw else "{}"


def destroy(h) -> None:
    _lib.vbx_destroy(h)


def abi_version() -> int:
    return int(_lib.vbx_abi_version())


def lib_version() -> str:
    raw = _lib.vbx_version_string()
    return raw.decode() if raw else ""
