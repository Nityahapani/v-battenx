import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.training import train, cv


def make_data(n=200, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = X @ w
    return X, y


def test_train_returns_booster():
    from vbatten_x.core import Booster
    X, y = make_data()
    b = train(X, y, num_boost_round=10)
    assert isinstance(b, Booster)
    assert b.num_stages == 10


def test_train_rmse_better_than_mean():
    X, y = make_data(n=300)
    b = train(X, y, num_boost_round=50, params={"learning_rate": 0.1})
    preds = b.predict(X)
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
        assert np.isfinite(v)
        assert v > 0
