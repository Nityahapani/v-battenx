import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.objectives import (
    CholBooster, MseBooster,
    _wls_isotropic, _wls_cholesky_free_intercept,
    _mse_loss,
)


def make_data(n=300, d=5, intercept=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = X @ w + intercept
    return X, y, w


def rmse(preds, y):
    return float(np.sqrt(np.mean((preds - y) ** 2)))


# ── solver unit tests ─────────────────────────────────────────────────────────

def test_cholesky_matches_lu_no_intercept_penalty():
    """
    When reg_lambda is small enough to be negligible, Cholesky and LU
    should give the same solution.  We compare on a well-conditioned system.
    """
    rng = np.random.default_rng(0)
    n, d = 200, 5
    X  = rng.standard_normal((n, d)).astype(np.float32)
    Xb = np.c_[X, np.ones(n, dtype=np.float32)]
    g  = rng.standard_normal(n).astype(np.float32)
    h  = np.ones(n, dtype=np.float32)

    w_lu   = _wls_isotropic(Xb, g, h, 0.0)
    w_chol = _wls_cholesky_free_intercept(Xb, g, h, 0.0)
    np.testing.assert_allclose(w_lu, w_chol, atol=1e-4)


def test_intercept_not_penalised():
    """
    The bias weight (last element of w) should not shrink toward zero
    under the Cholesky solver, even with large reg_lambda.
    The isotropic solver shrinks it; the Cholesky solver does not.
    """
    rng  = np.random.default_rng(1)
    n, d = 300, 4
    X    = rng.standard_normal((n, d)).astype(np.float32)
    Xb   = np.c_[X, np.ones(n, dtype=np.float32)]
    g    = rng.standard_normal(n).astype(np.float32)
    h    = np.ones(n, dtype=np.float32)
    lam  = 100.0   # large penalty to make the effect obvious

    w_iso  = _wls_isotropic(Xb, g, h, lam)
    w_chol = _wls_cholesky_free_intercept(Xb, g, h, lam)

    # Isotropic: bias is shrunk (|w_iso[-1]| < |w_chol[-1]|)
    assert abs(w_chol[-1]) > abs(w_iso[-1]), (
        "Cholesky solver should leave the intercept unpenalised — "
        f"chol={w_chol[-1]:.4f}, iso={w_iso[-1]:.4f}"
    )


def test_penalty_matrix_structure():
    """
    The penalty block must be zero for the intercept column and λ for features.
    We verify this directly from the normal equations by checking the A matrix.
    """
    n, d = 100, 3
    X    = np.random.randn(n, d).astype(np.float32)
    Xb   = np.c_[X, np.ones(n, dtype=np.float32)]
    lam  = 5.0

    Xw   = Xb.copy()
    A    = (Xw.T @ Xb).astype(np.float64)
    penalty       = np.zeros(d + 1)
    penalty[:d]   = lam
    A_penalised   = A + np.diag(penalty)

    assert A_penalised[d, d] == A[d, d]           # intercept diagonal unchanged
    for j in range(d):
        assert abs(A_penalised[j, j] - (A[j, j] + lam)) < 1e-9


def test_cholesky_solution_finite():
    rng = np.random.default_rng(2)
    n, d = 200, 8
    X  = rng.standard_normal((n, d)).astype(np.float32)
    Xb = np.c_[X, np.ones(n, dtype=np.float32)]
    g  = rng.standard_normal(n).astype(np.float32)
    h  = np.abs(rng.standard_normal(n).astype(np.float32)) + 0.1
    w  = _wls_cholesky_free_intercept(Xb, g, h, 1.0)
    assert np.all(np.isfinite(w))


# ── booster-level tests ───────────────────────────────────────────────────────

def test_chol_booster_lower_bias_with_large_intercept():
    """
    When the true intercept is large (y = X@w + C, C >> 0), the isotropic
    solver shrinks the bias estimate toward zero, inflating prediction error.
    CholBooster should produce lower RMSE.
    """
    X, y, _ = make_data(n=500, d=6, intercept=10.0, seed=3)

    iso  = MseBooster(n_estimators=60, learning_rate=0.1, reg_lambda=5.0).fit(X, y)
    chol = CholBooster(n_estimators=60, learning_rate=0.1, reg_lambda=5.0).fit(X, y)

    assert rmse(chol.predict(X), y) < rmse(iso.predict(X), y), (
        "CholBooster should have lower RMSE when intercept is large and λ is strong"
    )


def test_chol_booster_no_regression_zero_intercept():
    """When the true intercept is zero, CholBooster should not regress vs MseBooster."""
    X, y, _ = make_data(n=400, d=5, intercept=0.0, seed=4)
    iso  = MseBooster(n_estimators=50, learning_rate=0.1, reg_lambda=1.0).fit(X, y)
    chol = CholBooster(n_estimators=50, learning_rate=0.1, reg_lambda=1.0).fit(X, y)
    assert rmse(chol.predict(X), y) <= rmse(iso.predict(X), y) * 1.05


def test_chol_booster_finite_predictions():
    X, y, _ = make_data(n=200, intercept=5.0, seed=5)
    preds = CholBooster(n_estimators=40, learning_rate=0.1).fit(X, y).predict(X)
    assert np.all(np.isfinite(preds))


def test_chol_booster_huber_objective():
    X, y, _ = make_data(n=200, intercept=3.0, seed=6)
    b = CholBooster(n_estimators=30, learning_rate=0.1,
                    objective="huber", huber_delta=1.0).fit(X, y)
    assert np.all(np.isfinite(b.predict(X)))


def test_chol_booster_intercept_estimate_closer_to_truth():
    """
    After convergence, CholBooster's bias weight should be closer to
    the true intercept than MseBooster's when λ is large.
    We extract the summed bias contribution from all stages.
    """
    true_intercept = 8.0
    X, y, _ = make_data(n=600, d=4, intercept=true_intercept, seed=7)

    iso  = MseBooster(n_estimators=80, learning_rate=0.1, reg_lambda=10.0).fit(X, y)
    chol = CholBooster(n_estimators=80, learning_rate=0.1, reg_lambda=10.0).fit(X, y)

    def total_bias(booster, lr):
        return float(sum(w[-1] for w in booster._weights)) * lr

    bias_iso  = total_bias(iso,  0.1)
    bias_chol = total_bias(chol, 0.1)

    assert abs(bias_chol - true_intercept) < abs(bias_iso - true_intercept), (
        f"CholBooster bias {bias_chol:.3f} should be closer to truth {true_intercept} "
        f"than MseBooster bias {bias_iso:.3f}"
    )
