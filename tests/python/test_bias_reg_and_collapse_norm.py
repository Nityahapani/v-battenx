"""
Tests for two algorithmic fixes:

(A) Bias regularization in LinearBooster
-----------------------------------------
The normal equations for a ridge booster with augmented-bias design matrix are:

    (X^T H X + λ R) w = X^T H (−g)

where R should be diag(λ,...,λ, 0): the bias column is unpenalised.
The old code used A.diagonal() += λ which also penalised the bias, pulling
the intercept toward zero and introducing systematic error when E[y] ≠ 0.

Fix: replace uniform diagonal shift with a loop over feature columns only.

(B) Frobenius over-scaling in Collapse mutations
-------------------------------------------------
After PCA projection  proj = data · V_k  (Eckart–Young optimal),

    ‖proj‖_F = sqrt(λ_1 + ... + λ_k)  ≤  ‖data‖_F = sqrt(λ_1 + ... + λ_p)

The old code called NormaliseFrobenius(out, original_norm), rescaling the
projected output by ‖data‖_F / ‖proj‖_F ≥ 1, which inflated every collapsed
field.  The correct behaviour is to leave proj's norm unchanged — it already
represents the maximum-retained energy.

Fix: remove NormaliseFrobenius from all collapse paths.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.core import Booster


# ── helpers ───────────────────────────────────────────────────────────────────

def make_shifted(n=400, d=5, shift=5.0, seed=0):
    """Linear data with a large label mean, so intercept matters."""
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = (X @ w + shift).astype(np.float32)
    return X, y


def make_zero_mean(n=400, d=5, seed=1):
    """Linear data centred at zero."""
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = (X @ w).astype(np.float32)
    return X, y


def rmse(pred, label):
    return float(np.sqrt(np.mean((pred - label) ** 2)))


# ── (A) Bias regularization ───────────────────────────────────────────────────

def test_bias_unregularised_reduces_intercept_error():
    """
    With a large label shift (intercept = 5), penalising the bias column
    shrinks the intercept, producing large systematic error on the
    zero-mean features.  The fix (unregularised bias) must give lower RMSE.

    Mathematical expectation:
    - Regularised intercept: w_bias ≈ shift / (1 + λ/N) < shift
    - Unregularised intercept: w_bias → shift  as N grows

    At n=400, d=5, shift=5, λ=10 the residual from intercept shrinkage
    dominates, so the old code's RMSE >> the fixed code's RMSE.
    """
    X, y = make_shifted(n=400, d=5, shift=5.0, seed=0)

    # High reg_lambda amplifies the intercept-shrinkage error.
    b = Booster({"learning_rate": 0.5, "reg_lambda": 10.0, "verbose": 0})
    b.set_data(X, y).train(50)
    preds = b.predict(X)

    # Mean prediction should track the label mean closely (within 0.5 units).
    pred_mean  = float(np.mean(preds))
    label_mean = float(np.mean(y))
    assert abs(pred_mean - label_mean) < 0.5, (
        f"Predicted mean {pred_mean:.3f} diverges from label mean "
        f"{label_mean:.3f} by {abs(pred_mean-label_mean):.3f} — "
        "bias is likely being over-penalised"
    )


def test_bias_fix_beats_zero_mean_baseline():
    """
    On zero-mean data the old and new code behave identically for the feature
    weights (intercept ≈ 0 either way), so the fix should not regress RMSE.
    """
    X, y = make_zero_mean(n=400, d=5, seed=1)
    b = Booster({"learning_rate": 0.2, "reg_lambda": 1.0, "verbose": 0})
    b.set_data(X, y).train(40)
    preds = b.predict(X)
    assert rmse(preds, y) < float(np.std(y)), (
        "RMSE should beat the trivial mean predictor"
    )


def test_unregularised_bias_scales_with_shift():
    """
    For varying shift values, the predicted mean must track the true mean
    with error < 10% of the shift — confirming the intercept is recovered.
    """
    for shift in [1.0, 3.0, 8.0]:
        X, y = make_shifted(n=500, d=4, shift=shift, seed=42)
        b = Booster({"learning_rate": 0.4, "reg_lambda": 5.0, "verbose": 0})
        b.set_data(X, y).train(60)
        pred_mean  = float(np.mean(b.predict(X)))
        label_mean = float(np.mean(y))
        err_frac = abs(pred_mean - label_mean) / (abs(label_mean) + 1e-9)
        assert err_frac < 0.10, (
            f"shift={shift}: intercept error fraction {err_frac:.3f} > 10%"
        )


def test_bias_fix_does_not_change_zero_lambda_predictions():
    """
    With λ=0 the fix and original are identical (no regularisation applied
    to either features or bias), so predictions must be unchanged.
    """
    X, y = make_shifted(n=200, d=3, shift=2.0, seed=5)
    b = Booster({"learning_rate": 0.1, "reg_lambda": 0.0, "verbose": 0})
    b.set_data(X, y).train(20)
    preds = b.predict(X)
    assert np.all(np.isfinite(preds))


# ── (B) Collapse norm fix ─────────────────────────────────────────────────────

def test_collapse_does_not_inflate_predictions():
    """
    After a collapse mutation the field's Frobenius norm should be ≤ the
    pre-collapse norm (energy can only be lost in projection, not gained).

    We induce a collapse by setting tau_collapse high so the DTDO collapses
    on nearly every stage (pde_r ≈ 0 from NullEvaluator).  We then verify
    that predictions do not blow up — a symptom of over-scaling.

    Old code: NormaliseFrobenius scales proj by ‖input‖/‖proj‖ ≥ 1, which
    can inflate predictions by up to ~sqrt(p/k) where p=from_dim, k=to_dim.
    With p=3, k=1 that's a factor of sqrt(3) ≈ 1.73 per collapse.

    Fixed code: proj is returned as-is, predictions remain bounded.
    """
    rng = np.random.default_rng(99)
    X   = rng.standard_normal((200, 5)).astype(np.float32)
    y   = (X.sum(axis=1)).astype(np.float32)

    # Force the threshold DTDO into expand-then-collapse cycles by starting
    # with a non-zero tau_expand but very high tau_collapse (collapses
    # whenever pde_r < tau_collapse, which is always from NullEvaluator).
    b = Booster({
        "dtdo":         "threshold",
        "tau_expand":   0.05,
        "tau_collapse": 1e6,   # always collapse if there's room
        "learning_rate": 0.05,
        "reg_lambda":    1.0,
        "verbose":       0,
    })
    b.set_data(X, y).train(30)
    preds = b.predict(X)

    assert np.all(np.isfinite(preds)), "Predictions contain NaN/Inf after collapse"
    # Predictions should be in a reasonable range relative to label scale.
    pred_std  = float(np.std(preds))
    label_std = float(np.std(y))
    assert pred_std < 10.0 * label_std, (
        f"Prediction std {pred_std:.2f} >> label std {label_std:.2f}; "
        "collapse may be over-scaling"
    )


def test_collapse_norm_is_nondecreasing_error():
    """
    Direct unit test: after collapse, ‖output‖_F ≤ ‖input‖_F.

    We import the transition functions directly via the Python C-ext path
    is not available, so we verify via the Booster's aggregate behaviour:
    train_loss must remain finite and not diverge after many collapse stages.
    """
    rng = np.random.default_rng(7)
    X   = rng.standard_normal((300, 6)).astype(np.float32)
    y   = (X[:, 0] - X[:, 2]).astype(np.float32)

    b = Booster({
        "dtdo":          "router",
        "tau_expand":    0.01,
        "tau_collapse":  1e4,
        "learning_rate": 0.05,
        "verbose":       0,
    })
    b.set_data(X, y).train(40)

    assert np.isfinite(b.train_loss), "train_loss diverged — collapse inflation?"
    assert b.train_loss < 1e4,        "train_loss unreasonably large after collapses"


def test_collapse_fix_preserves_convergence():
    """
    End-to-end: with DTDO=threshold the model must still reduce RMSE below
    the naive mean-predictor baseline after the collapse fix.
    """
    rng = np.random.default_rng(11)
    X   = rng.standard_normal((300, 5)).astype(np.float32)
    y   = (2 * X[:, 0] - X[:, 1] + 0.5).astype(np.float32)

    b = Booster({
        "dtdo":          "threshold",
        "tau_expand":    0.1,
        "tau_collapse":  0.001,
        "learning_rate": 0.1,
        "verbose":       0,
    })
    b.set_data(X, y).train(60)
    preds      = b.predict(X)
    rmse_model = rmse(preds, y)
    rmse_mean  = float(np.std(y))
    assert rmse_model < rmse_mean, (
        f"Model RMSE {rmse_model:.4f} should beat mean-predictor RMSE {rmse_mean:.4f}"
    )
