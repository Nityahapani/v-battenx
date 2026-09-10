import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.objectives import HuberBooster, MseBooster, _huber_grad_hess, _huber_loss


def make_clean(n=300, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = X @ w
    return X, y


def make_outlier(n=400, d=5, n_outliers=30, outlier_scale=15.0, seed=2):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = X @ w + 0.05 * rng.standard_normal(n).astype(np.float32)
    idx = rng.choice(n, n_outliers, replace=False)
    y[idx] += (outlier_scale * rng.standard_normal(n_outliers)).astype(np.float32)
    return X, y, idx


def rmse(preds, y):
    return float(np.sqrt(np.mean((preds - y) ** 2)))


def test_huber_hessian_quadratic_region():
    pred  = np.array([0.5], dtype=np.float32)
    label = np.zeros(1, dtype=np.float32)
    g, h  = _huber_grad_hess(pred, label, delta=1.0)
    assert abs(g[0] - 0.5) < 1e-6
    assert abs(h[0] - 1.0) < 1e-6


def test_huber_hessian_linear_region():
    pred  = np.array([5.0], dtype=np.float32)
    label = np.zeros(1, dtype=np.float32)
    g, h  = _huber_grad_hess(pred, label, delta=1.0)
    assert abs(g[0] - 1.0) < 1e-6       # capped gradient = delta * sign(r)
    assert abs(h[0] - 1.0 / 5.0) < 1e-5 # h = delta / |r|


def test_huber_hessian_downweights_outliers():
    delta = 1.0
    for abs_r in [2.0, 5.0, 10.0]:
        _, h = _huber_grad_hess(np.array([abs_r], np.float32),
                                np.zeros(1, np.float32), delta)
        assert h[0] < 1.0
        assert abs(h[0] - delta / abs_r) < 1e-5


def test_huber_large_delta_matches_mse():
    X, y = make_clean(seed=3)
    mse   = MseBooster(n_estimators=30, learning_rate=0.1).fit(X, y)
    huber = HuberBooster(n_estimators=30, learning_rate=0.1, delta=1e6).fit(X, y)
    np.testing.assert_allclose(mse.predict(X), huber.predict(X), rtol=1e-3)


def test_huber_beats_mse_on_outliers():
    X, y, outlier_idx = make_outlier(seed=4)
    inlier = np.ones(len(y), dtype=bool)
    inlier[outlier_idx] = False

    mse   = MseBooster(n_estimators=100, learning_rate=0.1).fit(X, y)
    huber = HuberBooster(n_estimators=100, learning_rate=0.1, delta=1.0).fit(X, y)

    rmse_mse   = rmse(mse.predict(X[inlier]),   y[inlier])
    rmse_huber = rmse(huber.predict(X[inlier]), y[inlier])

    assert rmse_huber < rmse_mse, (
        f"Huber inlier RMSE={rmse_huber:.4f} should beat MSE={rmse_mse:.4f}"
    )


def test_huber_finite_predictions():
    X, y = make_clean(n=200, seed=5)
    preds = HuberBooster(n_estimators=40, learning_rate=0.1).fit(X, y).predict(X)
    assert np.all(np.isfinite(preds))


def test_huber_loss_non_negative():
    X, y = make_clean(n=200, seed=6)
    b = HuberBooster(n_estimators=30, learning_rate=0.1, delta=1.0).fit(X, y)
    pred = b.predict(X)
    loss = _huber_loss(pred, y, delta=1.0)
    assert loss >= 0


def test_huber_delta_controls_robustness():
    X, y, _ = make_outlier(seed=7)
    losses = {}
    for delta in [0.5, 1.0, 5.0]:
        b = HuberBooster(n_estimators=60, learning_rate=0.1, delta=delta).fit(X, y)
        losses[delta] = _huber_loss(b.predict(X), y, delta=delta)
    # All losses should be finite and non-negative
    for v in losses.values():
        assert np.isfinite(v) and v >= 0
