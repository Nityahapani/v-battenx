import ctypes
import os
import numpy as np

_here = os.path.dirname(__file__)
_lib  = ctypes.CDLL(os.path.join(_here, "_vbatten_x.so"))

_lib.vbx_learner_create.restype  = ctypes.c_void_p
_lib.vbx_learner_create.argtypes = [ctypes.c_char_p]

_lib.vbx_set_data.restype  = ctypes.c_int
_lib.vbx_set_data.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_long, ctypes.c_long,
]

_lib.vbx_train.restype  = ctypes.c_int
_lib.vbx_train.argtypes = [ctypes.c_void_p, ctypes.c_int]

_lib.vbx_predict.restype  = ctypes.c_int
_lib.vbx_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_long, ctypes.c_long,
    ctypes.POINTER(ctypes.c_float),
]

_lib.vbx_save.restype  = ctypes.c_int
_lib.vbx_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.vbx_load.restype  = ctypes.c_int
_lib.vbx_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

_lib.vbx_train_loss.restype  = ctypes.c_float
_lib.vbx_train_loss.argtypes = [ctypes.c_void_p]

_lib.vbx_num_stages.restype  = ctypes.c_int
_lib.vbx_num_stages.argtypes = [ctypes.c_void_p]

_lib.vbx_destroy.restype  = None
_lib.vbx_destroy.argtypes = [ctypes.c_void_p]

_lib.vbx_last_error.restype  = ctypes.c_char_p
_lib.vbx_last_error.argtypes = []


def _check(ret: int) -> None:
    if ret != 0:
        raise RuntimeError(_lib.vbx_last_error().decode())


def _f32(arr):
    return np.asarray(arr, dtype=np.float32, order="C")


def create(params_json: str) -> ctypes.c_void_p:
    h = _lib.vbx_learner_create(params_json.encode())
    if h is None:
        raise RuntimeError(_lib.vbx_last_error().decode())
    return h


def set_data(h, X, y):
    X_ = _f32(X)
    y_ = _f32(y)
    nrows, ncols = X_.shape
    _check(_lib.vbx_set_data(
        h,
        X_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nrows, ncols,
    ))


def train(h, n_iters: int):
    _check(_lib.vbx_train(h, n_iters))


def predict(h, X) -> np.ndarray:
    X_ = _f32(X)
    nrows, ncols = X_.shape
    out = np.zeros(nrows, dtype=np.float32)
    _check(_lib.vbx_predict(
        h,
        X_.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        nrows, ncols,
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ))
    return out


def save(h, path: str):
    _check(_lib.vbx_save(h, path.encode()))


def load(h, path: str):
    _check(_lib.vbx_load(h, path.encode()))


def train_loss(h) -> float:
    return float(_lib.vbx_train_loss(h))


def num_stages(h) -> int:
    return int(_lib.vbx_num_stages(h))


def destroy(h):
    _lib.vbx_destroy(h)
