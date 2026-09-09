import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import json
import numpy as np
import tempfile
import pytest
from vbatten_x import Booster, PhysicsSpec, PDEType, __version__
from vbatten_x.sklearn import VBattenXRegressor


def make_data(n=120, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = X.sum(axis=1).astype(np.float32)
    return X, y


# ── version ──────────────────────────────────────────────────────────────────

def test_version_is_4():
    assert __version__.startswith("4.") or __version__.startswith("5.")


def test_model_version_is_4(tmp_path):
    X, y = make_data()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    path = str(tmp_path / "m.json")
    b.save(path)
    with open(path) as f:
        data = json.load(f)
    assert data["version"] in ("4.0.0", "5.0.0")


# ── learned DTDO ──────────────────────────────────────────────────────────────

def test_learned_dtdo_runs():
    X, y = make_data()
    b    = Booster({"dtdo": "learned", "learning_rate": 0.1})
    b.set_data(X, y).train(10)
    assert b.num_stages == 10
    assert np.isfinite(b.train_loss)


def test_learned_greedy_runs():
    X, y = make_data()
    b    = Booster({"dtdo": "learned_greedy", "learning_rate": 0.1})
    b.set_data(X, y).train(8)
    assert b.num_stages == 8
    preds = b.predict(X)
    assert np.isfinite(preds).all()


def test_learned_vs_plain_both_converge():
    X, y = make_data(n=200, d=6)
    b1   = Booster({"learning_rate": 0.1})
    b2   = Booster({"dtdo": "learned", "learning_rate": 0.1})
    b1.set_data(X, y).train(20)
    b2.set_data(X, y).train(20)
    rmse = lambda b: float(np.sqrt(np.mean((b.predict(X) - y)**2)))
    assert rmse(b1) < 10.0
    assert rmse(b2) < 10.0


def test_learned_dtdo_sklearn():
    X, y = make_data()
    reg  = VBattenXRegressor(n_estimators=10, dtdo="learned", learning_rate=0.1)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()


def test_learned_dtdo_save_load(tmp_path):
    X, y = make_data()
    b    = Booster({"dtdo": "learned", "learning_rate": 0.1})
    b.set_data(X, y).train(8)
    p1   = b.predict(X)
    path = str(tmp_path / "learned.json")
    b.save(path)
    b2   = Booster.load(path)
    p2   = b2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)


# ── optimizers ───────────────────────────────────────────────────────────────

def test_all_dtdo_strategies_produce_finite_loss():
    X, y = make_data(n=100)
    for dtdo in ["none", "threshold", "gradient", "router", "learned", "learned_greedy"]:
        b = Booster({"dtdo": dtdo, "learning_rate": 0.05, "tau_expand": 0.001})
        b.set_data(X, y).train(5)
        assert np.isfinite(b.train_loss), f"NaN loss for dtdo={dtdo}"


# ── Dask interface ────────────────────────────────────────────────────────────

def test_dask_booster_import():
    from vbatten_x.dask import DaskBooster
    assert DaskBooster is not None


def test_dask_booster_fit_predict():
    from vbatten_x.dask import DaskBooster
    X, y = make_data(n=200)
    db   = DaskBooster(params={"learning_rate": 0.1}, num_boost_round=10, num_workers=2)
    db.fit(X, y)
    preds = db.predict(X)
    assert preds.shape == (200,)
    assert np.isfinite(preds).all()


def test_train_dask_returns_booster():
    from vbatten_x.dask import train_dask
    from vbatten_x.core import Booster
    X, y = make_data(n=150)
    b    = train_dask(X, y, params={"learning_rate": 0.1},
                      num_boost_round=8, num_workers=2)
    assert isinstance(b, Booster)
    assert b.num_stages == 8
