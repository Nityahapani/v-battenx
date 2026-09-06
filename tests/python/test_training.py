import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.training import train, cv
from vbatten_x.core import Booster


def make_data(n=200, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = (X @ rng.standard_normal(d).astype(np.float32))
    return X, y


def test_train_returns_booster():
    X, y = make_data()
    b = train(X, y, num_boost_round=10)
    assert isinstance(b, Booster)
    assert b.num_stages == 10


def test_train_rmse_better_than_mean():
    X, y = make_data(n=300)
    b    = train(X, y, num_boost_round=50, params={"learning_rate": 0.1})
    preds     = b.predict(X)
    rmse_model = np.sqrt(np.mean((preds - y) ** 2))
    rmse_mean  = np.sqrt(np.mean((y.mean() - y) ** 2))
    assert rmse_model < rmse_mean


def test_cv_returns_dict():
    X, y = make_data(n=200)
    result = cv(X, y, nfold=3, num_boost_round=10)
    assert "train-rmse-mean" in result
    assert "test-rmse-mean"  in result
    assert len(result["train-rmse-mean"]) == 3


def test_cv_val_rmse_finite():
    X, y = make_data(n=250, d=4)
    result = cv(X, y, nfold=4, num_boost_round=15)
    for v in result["test-rmse-mean"]:
        assert np.isfinite(v) and v > 0


def test_dtdo_threshold_runs():
    X, y = make_data(n=150)
    b = Booster({"dtdo": "threshold", "tau_expand": 0.01, "learning_rate": 0.1})
    b.set_data(X, y).train(20)
    assert b.num_stages == 20
    assert np.isfinite(b.train_loss)


def test_dtdo_router_runs():
    X, y = make_data(n=150)
    b = Booster({"dtdo": "router", "learning_rate": 0.1, "tau_expand": 0.05})
    b.set_data(X, y).train(15)
    assert b.num_stages == 15
    assert np.isfinite(b.train_loss)


def test_dtdo_none_same_as_plain():
    X, y = make_data(n=100)
    b1 = Booster({"learning_rate": 0.1})
    b2 = Booster({"learning_rate": 0.1, "dtdo": "none"})
    b1.set_data(X, y).train(10)
    b2.set_data(X, y).train(10)
    np.testing.assert_allclose(b1.predict(X), b2.predict(X), atol=1e-5)
