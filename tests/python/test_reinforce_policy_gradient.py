"""
Tests for the corrected REINFORCE policy gradient in DtdoTrainer.

Background
----------
The previous Update() computed:

    grad_scale = -lr * pg_loss
    W *= (1 - grad_scale * 1e-4)          # scalar weight decay on ALL weights

This is not a policy gradient. A correct REINFORCE update (Williams 1992) is:

    ∇_θ J(θ) = E_π[ ∇_θ log π_θ(a|s) · Â(s,a) ]

where Â = (r - mean_r) / std_r is the normalised advantage.

For the two-layer head  h = GELU(W1·f + b1),  logit = W2·h + b2:

    δ_logit = (π - e_a) · (−Â) / B           [softmax CE gradient]
    ∂W2 += δ_logit · hᵀ
    ∂b2 += δ_logit
    δ_h    = W2ᵀ · δ_logit
    δ_pre  = δ_h ⊙ GELU'(pre)
    ∂W1 += δ_pre · fᵀ
    ∂b1 += δ_pre

The old code applied the SAME scalar multiplier to every weight regardless of
which action was taken or what the gradient direction is — it cannot
differentiate between actions and therefore cannot learn a policy.

Tests
-----
1. GELU derivative correctness (finite-difference check).
2. Imitation step pulls the target action logit up and others down.
3. After repeated REINFORCE updates with a fixed reward signal (one action
   consistently rewarded), the probability of that action strictly increases.
4. Convergence: after 500 updates favouring a fixed action, P(correct) > 0.5.
5. No divergence: weights stay finite after many updates.
6. Learned DTDO runs end-to-end without NaN in train_loss.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x.core import Booster


# ── helper: access the learned DTDO through the Python Booster ────────────────
# The trainer lives in C++; we test it indirectly by checking that the
# learned DTDO converges on a synthetic task where one mutation is always
# rewarded (we proxy this by checking train_loss on a structured dataset)
# and directly via a pure-Python reimplementation of the math.


# ── 1. GELU derivative (finite-difference) ───────────────────────────────────

def gelu(x):
    c = 0.7978845608
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))

def gelu_grad_analytical(x):
    """GELU'(x) as implemented in GeluGrad()."""
    c = 0.7978845608
    u  = c * (x + 0.044715 * x**3)
    t  = np.tanh(u)
    du = c * (1 + 3 * 0.044715 * x**2)
    return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * du

def test_gelu_derivative_finite_difference():
    """Analytical GELU' must match the finite-difference approximation."""
    rng = np.random.default_rng(0)
    xs  = rng.uniform(-3, 3, 200)
    eps = 1e-4
    for x in xs:
        fd   = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
        anal = gelu_grad_analytical(x)
        assert abs(fd - anal) < 1e-4, f"x={x:.3f}: fd={fd:.6f} anal={anal:.6f}"


# ── 2. Softmax cross-entropy gradient ────────────────────────────────────────

def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()

def pg_gradient(logits, action_idx, advantage):
    """
    Gradient of L = -log π(a) · Â w.r.t. logits.
    d(log π(a))/d(logits) = e_a - π  (standard softmax result)
    dL/d(logits) = -Â · (e_a - π) = Â · (π - e_a)
    Gradient descent: logits -= lr · Â · (π - e_a)
    When Â > 0: (π[a] - 1) < 0, so logits[a] increases → P(a) increases. ✓
    """
    pi        = softmax(logits)
    delta     = pi.copy()
    delta[action_idx] -= 1.0
    return delta * advantage   # ∂L/∂logits = Â·(π - e_a)

def test_pg_gradient_direction():
    """
    When advantage > 0, the gradient should DECREASE the chosen action's
    logit (because we subtract the gradient in a gradient descent step,
    which increases the log-prob, which increases the probability of the
    chosen action).
    """
    rng     = np.random.default_rng(1)
    logits  = rng.standard_normal(8).astype(np.float32)
    action  = 3
    adv     = 1.0   # positive: action was good

    delta = pg_gradient(logits, action, adv)
    # gradient descent: logits_new = logits - lr * delta
    lr          = 0.1
    logits_new  = logits - lr * delta
    pi_old      = softmax(logits)
    pi_new      = softmax(logits_new)

    assert pi_new[action] > pi_old[action], (
        f"After positive-advantage update, P(a={action}) should increase: "
        f"{pi_old[action]:.4f} → {pi_new[action]:.4f}"
    )


def test_pg_gradient_negative_advantage():
    """When advantage < 0 the chosen action's probability should decrease."""
    rng    = np.random.default_rng(2)
    logits = rng.standard_normal(8).astype(np.float32)
    action = 5
    adv    = -1.0

    delta      = pg_gradient(logits, action, adv)
    logits_new = logits - 0.1 * delta
    pi_old     = softmax(logits)
    pi_new     = softmax(logits_new)

    assert pi_new[action] < pi_old[action], (
        f"After negative-advantage update, P(a={action}) should decrease: "
        f"{pi_old[action]:.4f} → {pi_new[action]:.4f}"
    )


# ── 3. Full backprop through two-layer head ───────────────────────────────────

class TwoLayerHead:
    """Pure-Python reimplementation of DtdoNet's head for unit-testing."""
    def __init__(self, fd=8, hd=16, od=7, seed=0):
        rng   = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / fd)
        self.W1 = rng.standard_normal((hd, fd)).astype(np.float32) * scale
        self.b1 = np.zeros(hd, dtype=np.float32)
        scale2  = np.sqrt(2.0 / hd)
        self.W2 = rng.standard_normal((od, hd)).astype(np.float32) * scale2
        self.b2 = np.zeros(od, dtype=np.float32)

    def forward(self, f):
        pre    = self.W1 @ f + self.b1
        h      = gelu(pre)
        logits = self.W2 @ h + self.b2
        return pre, h, logits

    def reinforce_step(self, f, pre, h, action_idx, advantage, lr, batch_size=1):
        pi          = softmax(self.W2 @ h + self.b2)  # use current W2
        delta_logit = pi.copy()
        delta_logit[action_idx] -= 1.0
        delta_logit *= advantage / batch_size   # ∂L/∂logits = Â·(π - e_a)

        self.W2 -= lr * np.outer(delta_logit, h)
        self.b2 -= lr * delta_logit

        delta_h   = self.W2.T @ delta_logit
        delta_pre = delta_h * gelu_grad_analytical(pre)
        self.W1  -= lr * np.outer(delta_pre, f)
        self.b1  -= lr * delta_pre


def test_reinforce_increases_chosen_action_probability():
    """
    Repeat REINFORCE update 200× with the same action rewarded (adv=+1).
    P(action) must strictly increase overall.
    """
    head   = TwoLayerHead(fd=8, hd=16, od=7, seed=3)
    rng    = np.random.default_rng(3)
    f      = rng.standard_normal(8).astype(np.float32)
    action = 2
    lr     = 0.05

    pre0, h0, logits0 = head.forward(f)
    p0 = softmax(logits0)[action]

    for _ in range(200):
        pre, h, _ = head.forward(f)
        head.reinforce_step(f, pre, h, action, advantage=1.0, lr=lr)

    _, _, logits_final = head.forward(f)
    p_final = softmax(logits_final)[action]

    assert p_final > p0, (
        f"P(action={action}) should increase: {p0:.4f} → {p_final:.4f}"
    )


def test_reinforce_convergence_to_majority():
    """
    After 500 updates always rewarding action 4, P(4) should exceed 0.5.
    This is provably impossible with the old scalar-multiplicative update.
    """
    head   = TwoLayerHead(fd=8, hd=32, od=9, seed=7)
    rng    = np.random.default_rng(7)
    f      = rng.standard_normal(8).astype(np.float32)
    action = 4
    lr     = 0.02

    for _ in range(500):
        pre, h, _ = head.forward(f)
        head.reinforce_step(f, pre, h, action, advantage=1.0, lr=lr)

    _, _, logits_final = head.forward(f)
    p = softmax(logits_final)[action]

    assert p > 0.5, (
        f"After 500 positive-advantage updates, P(action=4) = {p:.4f}, expected > 0.5"
    )


def test_weights_stay_finite_after_many_updates():
    """
    Neither W1 nor W2 should diverge under repeated REINFORCE updates with
    mixed rewards.  Old code's multiplicative update could diverge when
    pg_loss < 0  (since 1 - c * 1e-4 < 1 could flip sign after enough steps).
    """
    head = TwoLayerHead(fd=8, hd=16, od=7, seed=9)
    rng  = np.random.default_rng(9)

    for _ in range(1000):
        f      = rng.standard_normal(8).astype(np.float32)
        action = int(rng.integers(0, 7))
        adv    = float(rng.choice([-1.0, 1.0]))
        pre, h, _ = head.forward(f)
        head.reinforce_step(f, pre, h, action, adv, lr=0.01)

    assert np.all(np.isfinite(head.W1)), "W1 contains NaN/Inf"
    assert np.all(np.isfinite(head.W2)), "W2 contains NaN/Inf"


# ── 4. End-to-end: learned DTDO runs without NaN ─────────────────────────────

def test_learned_dtdo_runs():
    rng = np.random.default_rng(42)
    X   = rng.standard_normal((200, 5)).astype(np.float32)
    y   = (X[:, 0] + X[:, 1]).astype(np.float32)

    b = Booster({
        "dtdo":          "learned",
        "learning_rate": 0.05,
        "reg_lambda":    1.0,
        "verbose":       0,
    })
    b.set_data(X, y).train(30)
    preds = b.predict(X)

    assert np.all(np.isfinite(preds)), "Predictions contain NaN/Inf"
    assert np.isfinite(b.train_loss),  "train_loss is not finite"


def test_learned_greedy_dtdo_runs():
    rng = np.random.default_rng(43)
    X   = rng.standard_normal((200, 4)).astype(np.float32)
    y   = X.sum(axis=1).astype(np.float32)

    b = Booster({
        "dtdo":          "learned_greedy",
        "learning_rate": 0.05,
        "verbose":       0,
    })
    b.set_data(X, y).train(20)
    assert np.isfinite(b.train_loss)
