import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.core import Booster


def make_regression(n=200, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = X @ rng.standard_normal(d).astype(np.float32) + 0.1 * rng.standard_normal(n).astype(np.float32)
    return X, y


def test_booster_constructs():
    b = Booster({"objective": "regression"})
    assert b.num_stages == 0


def test_set_data_and_train():
    X, y = make_regression()
    b = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(20)
    assert b.num_stages == 20


def test_predict_shape():
    X, y = make_regression(n=100, d=4)
    b = Booster()
    b.set_data(X, y).train(10)
    preds = b.predict(X)
    assert preds.shape == (100,)


def test_predict_is_float32():
    X, y = make_regression()
    b = Booster()
    b.set_data(X, y).train(5)
    preds = b.predict(X)
    assert preds.dtype == np.float32


def test_train_loss_decreases():
    X, y = make_regression(n=300, d=6)
    losses = []
    for n_iters in [5, 20, 50]:
        b = Booster({"learning_rate": 0.05})
        b.set_data(X, y).train(n_iters)
        losses.append(b.train_loss)
    assert losses[0] >= losses[1] >= losses[2]


def test_save_load_roundtrip(tmp_path):
    X, y = make_regression(n=150)
    b = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(15)
    preds_before = b.predict(X)

    path = str(tmp_path / "model.json")
    b.save(path)

    b2 = Booster.load(path)
    preds_after = b2.predict(X)

    np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5, atol=1e-5)


def test_deterministic_with_same_seed():
    X, y = make_regression(seed=7)
    p1 = Booster({"learning_rate": 0.1}).set_data(X, y).train(10).predict(X)
    p2 = Booster({"learning_rate": 0.1}).set_data(X, y).train(10).predict(X)
    np.testing.assert_array_equal(p1, p2)
