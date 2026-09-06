import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x import PhysicsSpec, PDEType
from vbatten_x.core import Booster


def make_regression(n=100, d=6, seed=1):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = (X ** 2).sum(axis=1).astype(np.float32)
    return X, y


def test_lambda_pde_zero_equals_plain():
    X, y = make_regression()
    b1 = Booster({"learning_rate": 0.1, "lambda_pde": 0.0})
    b2 = Booster({"learning_rate": 0.1})
    b1.set_data(X, y).train(10)
    b2.set_data(X, y).train(10)
    p1 = b1.predict(X)
    p2 = b2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)


def test_conserve_energy_spec_builds():
    spec = PhysicsSpec().conserve("energy").conserve("mass")
    import json
    d = json.loads(spec.to_json())
    assert "energy" in d["conserved_quantities"]
    assert "mass"   in d["conserved_quantities"]


def test_training_with_conservation_spec_runs():
    X, y = make_regression(n=80)
    spec = PhysicsSpec().conserve("energy")
    b = Booster({"learning_rate": 0.05, "lambda_pde": 0.001})
    b.set_data(X, y).set_physics(spec).train(15)
    assert b.num_stages == 15
    assert np.isfinite(b.train_loss)


def test_physics_loss_is_finite_across_stages():
    X, y = make_regression(n=120)
    spec = PhysicsSpec().pde(PDEType.HEAT).grid(6, 6)
    b = Booster({"learning_rate": 0.05, "lambda_pde": 0.01})
    b.set_data(X, y).set_physics(spec).train(20)
    assert np.isfinite(b.train_loss)
    assert b.num_stages == 20
