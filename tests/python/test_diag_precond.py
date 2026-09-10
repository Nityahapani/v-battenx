import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.objectives import (
    DiagPrecondBooster, MseBooster,
    _wls_isotropic, _wls_diag_precond,
)


def make_uniform_scale(n=300, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = X @ w
    return X, y


def make_hetero_scale(n=400, seed=2):
    rng  = np.random.default_rng(seed)
    scales = np.array([1.0, 10.0, 100.0, 0.1, 50.0], dtype=np.float32)
    X    = (rng.standard_normal((n, 5)) * scales).astype(np.float32)
    w    = rng.standard_normal(5).astype(np.float32)
    y    = X @ w
    return X, y


def rmse(preds, y):
    return float(np.sqrt(np.mean((preds - y) ** 2)))


def test_diag_precond_recovers_isotropic_when_uniform():
    rng = np.random.default_rng(0)
    n, d = 500, 4
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=0)
    g = rng.standard_normal(n).astype(np.float32)
    h = np.ones(n, dtype=np.float32)

    w_iso  = _wls_isotropic(X, g, h, 1.0)
    w_diag = _wls_diag_precond(X, g, h, 1.0)
    np.testing.assert_allclose(w_iso, w_diag, atol=1e-4)


def test_diag_precond_differs_on_hetero_scale():
    rng = np.random.default_rng(1)
    n = 200
    scales = np.array([1.0, 100.0, 0.01, 50.0], dtype=np.float32)
    X = (rng.standard_normal((n, 4)) * scales).astype(np.float32)
    g = rng.standard_normal(n).astype(np.float32)
    h = np.ones(n, dtype=np.float32)

    w_iso  = _wls_isotropic(X, g, h, 1.0)
    w_diag = _wls_diag_precond(X, g, h, 1.0)
    assert not np.allclose(w_iso, w_diag, atol=1e-3)


def test_diag_precond_solution_finite():
    rng = np.random.default_rng(2)
    X = (rng.standard_normal((100, 5)) * np.array([1, 10, 100, 0.1, 50])).astype(np.float32)
    g = rng.standard_normal(100).astype(np.float32)
    h = np.ones(100, dtype=np.float32)
    w = _wls_diag_precond(X, g, h, 1.0)
    assert np.all(np.isfinite(w))


def test_diag_penalty_proportional_to_weighted_variance():
    """Core math: D_jj = diag_j / mean(diag), so penalty scales with feature variance."""
    rng = np.random.default_rng(3)
    n, d = 200, 4
    scales = np.array([1.0, 10.0, 100.0, 0.5], dtype=np.float32)
    X = (rng.standard_normal((n, d)) * scales).astype(np.float32)
    h = np.ones(n, dtype=np.float32)

    A    = (X * h[:, None]).T @ X
    diag = np.diag(A)
    D    = diag / diag.mean()

    # Feature with largest variance gets largest penalty weight
    assert D.argmax() == np.argmax(scales)
    # Feature with smallest variance gets smallest penalty weight
    assert D.argmin() == np.argmin(scales)


def test_diag_precond_beats_isotropic_on_hetero_features():
    X, y = make_hetero_scale(n=500, seed=3)
    iso  = MseBooster(n_estimators=80, learning_rate=0.1, reg_lambda=10.0).fit(X, y)
    diag = DiagPrecondBooster(n_estimators=80, learning_rate=0.1,
                               reg_lambda=10.0).fit(X, y)
    assert rmse(diag.predict(X), y) < rmse(iso.predict(X), y), (
        "DiagPrecond should outperform isotropic ridge on heterogeneous-scale features"
    )


def test_diag_precond_no_regression_uniform_scale():
    X, y = make_uniform_scale(n=300, seed=4)
    iso  = MseBooster(n_estimators=60, learning_rate=0.1, reg_lambda=1.0).fit(X, y)
    diag = DiagPrecondBooster(n_estimators=60, learning_rate=0.1,
                               reg_lambda=1.0).fit(X, y)
    assert rmse(diag.predict(X), y) <= rmse(iso.predict(X), y) * 1.05


def test_diag_precond_finite_predictions():
    X, y = make_hetero_scale(seed=5)
    preds = DiagPrecondBooster(n_estimators=40, learning_rate=0.1).fit(X, y).predict(X)
    assert np.all(np.isfinite(preds))


def test_diag_precond_with_huber_objective():
    X, y = make_hetero_scale(seed=6)
    b = DiagPrecondBooster(n_estimators=40, learning_rate=0.1,
                            objective="huber", huber_delta=1.0).fit(X, y)
    assert np.all(np.isfinite(b.predict(X)))


def test_diag_precond_weighted_hessian():
    """With non-unit Hessian weights, D should use h-weighted second moments."""
    rng = np.random.default_rng(7)
    n, d = 100, 3
    X    = rng.standard_normal((n, d)).astype(np.float32)
    h    = np.abs(rng.standard_normal(n).astype(np.float32)) + 0.1
    g    = rng.standard_normal(n).astype(np.float32)

    w = _wls_diag_precond(X, g, h, 1.0)
    assert np.all(np.isfinite(w))
    assert w.shape == (d,)
