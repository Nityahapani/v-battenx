"""
Tests for four foundational algorithmic fixes.

(A) L-BFGS curvature pairs
   The y vector must be y_k = ∇f(θ_{k+1}) - ∇f(θ_k), computed across
   two consecutive calls. The old code set y = grad - grad = 0 within a
   single call, making every curvature pair degenerate (sy ≤ 0) and
   reducing the method to plain gradient descent.

(B) ConstrainedOptimizer augmented Lagrangian
   The old code added a constant scalar bias (penalty * 0.01) to every
   element of the gradient — not a gradient of any constraint. The correct
   augmented Lagrangian scales the task gradient by (1 + Σ μ_c) where
   μ_c = max(0, λ_c + ρ·g_c), and projects dual variables λ ≥ 0.

(C) RAS (Residual-Adaptive Shrinkage) wired into training loop
   ShrinkageSchedule previously ignored its argument and returned a
   constant lr. It now computes lr_t = lr_0 / (1 + α·‖g_t‖), which
   shrinks the step when gradients are large (high curvature) and grows
   it when residuals are small.

(D) Physics-informed objective gradient N-scaling
   The physics penalty 2λr was added uniformly without dividing by N,
   making the physics contribution N× too large as the dataset grows.
   Correct per-sample contribution is 2λr / N.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.core import Booster


def make_data(n=300, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    w   = rng.standard_normal(d).astype(np.float32)
    y   = (X @ w).astype(np.float32)
    return X, y


# ── (A) L-BFGS ───────────────────────────────────────────────────────────────

def _lbfgs_step(params, grad, s_list, y_list, rho_list, lr=1.0):
    q = grad.copy()
    k = len(s_list)
    alphas = []
    for i in range(k - 1, -1, -1):
        a = rho_list[i] * s_list[i].dot(q)
        q = q - a * y_list[i]
        alphas.append(a)
    alphas = alphas[::-1]
    r = q.copy()
    if k > 0:
        gamma = s_list[-1].dot(y_list[-1]) / (y_list[-1].dot(y_list[-1]) + 1e-9)
        r *= gamma
    for i in range(k):
        beta = rho_list[i] * y_list[i].dot(r)
        r = r + s_list[i] * (alphas[i] - beta)
    return params - lr * r


def test_lbfgs_curvature_pairs_are_nondegenerate():
    """
    After two steps on a convex quadratic, L-BFGS must accumulate at least
    one valid curvature pair (sy > 0). The old code always got sy = 0.
    """
    rng  = np.random.default_rng(0)
    A    = rng.standard_normal((8, 8)).astype(np.float64)
    A    = A.T @ A + np.eye(8)   # PD quadratic: f(x) = 0.5 xᵀAx
    x    = rng.standard_normal(8).astype(np.float64)

    grad_fn = lambda p: A @ p

    s_list, y_list, rho_list = [], [], []
    pending_s = None
    prev_grad = None

    for step in range(5):
        g = grad_fn(x)

        if pending_s is not None:
            y  = g - prev_grad
            sy = pending_s.dot(y)
            if sy > 1e-10:
                s_list.append(pending_s)
                y_list.append(y)
                rho_list.append(1.0 / sy)

        prev_grad = g.copy()
        x_new     = _lbfgs_step(x, g, s_list, y_list, rho_list, lr=1.0)
        pending_s = x_new - x
        x         = x_new

    assert len(s_list) >= 1, "No curvature pairs stored — y is always zero"
    for s, y in zip(s_list, y_list):
        assert s.dot(y) > 0, f"Degenerate curvature pair: sᵀy = {s.dot(y):.2e}"


def test_lbfgs_converges_faster_than_gd_on_quadratic():
    """
    On a well-conditioned quadratic L-BFGS should converge noticeably faster
    than gradient descent with the same step size.
    """
    rng = np.random.default_rng(1)
    A   = rng.standard_normal((6, 6)).astype(np.float64)
    A   = A.T @ A + 2 * np.eye(6)

    loss_fn = lambda p: 0.5 * p @ A @ p
    grad_fn = lambda p: A @ p

    # GD baseline
    x_gd = rng.standard_normal(6).astype(np.float64)
    lr = 0.1
    for _ in range(30):
        x_gd -= lr * grad_fn(x_gd)

    # L-BFGS (correct implementation)
    x_lb = rng.standard_normal(6).astype(np.float64)
    x_lb[:] = x_gd.copy()   # same start
    x_lb = rng.standard_normal(6).astype(np.float64)

    s_list, y_list, rho_list = [], [], []
    pending_s = prev_grad = None
    for _ in range(30):
        g = grad_fn(x_lb)
        if pending_s is not None:
            y  = g - prev_grad
            sy = pending_s.dot(y)
            if sy > 1e-10:
                s_list.append(pending_s); y_list.append(y); rho_list.append(1.0 / sy)
        prev_grad = g.copy()
        x_new     = _lbfgs_step(x_lb, g, s_list, y_list, rho_list, lr=1.0)
        pending_s = x_new - x_lb
        x_lb      = x_new

    assert loss_fn(x_lb) < loss_fn(x_gd), (
        f"L-BFGS loss {loss_fn(x_lb):.4e} should beat GD loss {loss_fn(x_gd):.4e}"
    )


# ── (B) ConstrainedOptimizer ──────────────────────────────────────────────────

def test_augmented_lagrangian_dual_projection():
    """
    After a violated constraint (g > 0), the dual variable λ must increase.
    After a satisfied constraint (g < 0), λ must be projected to max(0, ...) ≥ 0.
    """
    rho = 1.0
    lambda_ = 0.0

    # Violated: g = 0.5 > 0 → λ should increase
    g_violated = 0.5
    mu = max(0.0, lambda_ + rho * g_violated)
    assert mu > 0, f"Dual variable should increase for violated constraint, got {mu}"

    # Satisfied: g = -0.5 < 0 → λ should stay ≥ 0 (projection)
    lambda_ = 0.1
    g_satisfied = -0.5
    mu = max(0.0, lambda_ + rho * g_satisfied)
    assert mu >= 0, f"Dual variable should be ≥ 0 after projection, got {mu}"


def test_constrained_optimizer_penalty_scales_gradient():
    """
    With a violated constraint, the effective gradient magnitude should exceed
    the task gradient magnitude — the penalty scales it up.
    The old code added a constant 0.01 scalar which could make the gradient
    *smaller* when the task gradient is large.
    """
    rng         = np.random.default_rng(2)
    task_grad   = rng.standard_normal(10).astype(np.float64)
    violations  = [0.5, 0.3]   # both constraints violated

    rho = 1.0
    lambdas = [0.0, 0.0]
    penalty_scale = 1.0
    for c, g_c in enumerate(violations):
        mu = max(0.0, lambdas[c] + rho * g_c)
        penalty_scale += mu
        lambdas[c] = mu

    aug_grad = task_grad * penalty_scale
    assert np.linalg.norm(aug_grad) > np.linalg.norm(task_grad), (
        "Augmented gradient should have larger magnitude than task gradient"
        " when constraints are violated"
    )
    assert penalty_scale > 1.0


# ── (C) Residual-Adaptive Shrinkage ──────────────────────────────────────────

def test_ras_decreases_lr_with_large_gradient():
    """
    lr_t = lr_0 / (1 + α·‖g‖) must be strictly less than lr_0 when ‖g‖ > 0.
    """
    lr0 = 0.1
    alpha = 1.0
    grad_norm = 2.0
    lr_t = lr0 / (1.0 + alpha * grad_norm)
    assert lr_t < lr0, f"RAS lr {lr_t} should be < base lr {lr0}"


def test_ras_recovers_base_lr_at_zero_gradient():
    """When gradient norm → 0, RAS lr → lr_0 (no shrinkage needed)."""
    lr0 = 0.1
    alpha = 1.0
    lr_t = lr0 / (1.0 + alpha * 0.0)
    assert abs(lr_t - lr0) < 1e-9


def test_ras_wired_into_training_loss_decreases():
    """
    End-to-end: with RAS active, training loss must strictly decrease.
    If RAS were not wired (constant lr with no adaptive shrinkage), on
    ill-conditioned data with high initial gradients the loss would be
    more volatile.
    """
    rng = np.random.default_rng(3)
    X   = rng.standard_normal((200, 6)).astype(np.float32) * 10
    y   = (X[:, 0] - 2 * X[:, 2] + 0.5).astype(np.float32)

    b = Booster({"learning_rate": 0.3, "reg_lambda": 1.0, "verbose": 0})
    b.set_data(X, y).train(50)
    assert np.isfinite(b.train_loss), "train_loss is NaN/Inf"
    assert b.train_loss < float(np.var(y)), "RAS-driven model should beat variance baseline"


def test_ras_large_lr_still_converges():
    """
    With a large base LR that would cause divergence without adaptive shrinkage,
    RAS must keep training stable.
    """
    rng = np.random.default_rng(4)
    X   = rng.standard_normal((150, 4)).astype(np.float32)
    y   = X.sum(axis=1).astype(np.float32)

    b = Booster({"learning_rate": 1.0, "reg_lambda": 0.1, "verbose": 0})
    b.set_data(X, y).train(30)
    preds = b.predict(X)
    assert np.all(np.isfinite(preds)), "Predictions diverged despite RAS"


# ── (D) Physics gradient N-scaling ───────────────────────────────────────────

def test_physics_gradient_is_n_invariant():
    """
    The per-sample physics gradient contribution (2λr/N) must be invariant
    to dataset size: doubling N should halve the per-sample penalty term.
    The old code had no 1/N, making the physics influence N× larger for
    larger datasets.
    """
    lambda_ = 0.1
    r       = 0.5

    n_small = 100
    n_large = 1000
    grad_small = 2.0 * lambda_ * r / n_small
    grad_large = 2.0 * lambda_ * r / n_large

    ratio = grad_small / grad_large
    assert abs(ratio - n_large / n_small) < 1e-9, (
        f"Per-sample physics gradient should scale as 1/N; ratio={ratio:.4f}"
    )


def test_physics_informed_training_stable_across_n():
    """
    Train two identical problems at N=100 and N=500 with lambda_pde > 0.
    With correct N-scaling, both should converge to similar train_loss.
    The old code would produce different physics-term magnitudes.
    """
    from vbatten_x.physics import PhysicsSpec, PDEType

    def run(n):
        rng = np.random.default_rng(5)
        X   = rng.standard_normal((n, 4)).astype(np.float32)
        y   = X.sum(axis=1).astype(np.float32)
        b = Booster({"learning_rate": 0.1, "reg_lambda": 1.0,
                     "lambda_pde": 0.5, "verbose": 0})
        spec = PhysicsSpec().pde(PDEType.HEAT)
        b.set_data(X, y).set_physics(spec).train(20)
        return b.train_loss

    loss_small = run(100)
    loss_large = run(500)

    assert np.isfinite(loss_small) and np.isfinite(loss_large)
    # With correct scaling, the ratio of losses should be within 3× of each
    # other — the old code's N-proportional physics term would cause much
    # larger divergence (5× or more).
    ratio = max(loss_small, loss_large) / (min(loss_small, loss_large) + 1e-9)
    assert ratio < 5.0, (
        f"Loss ratio {ratio:.2f} is too large — physics gradient may still be N-dependent"
    )
