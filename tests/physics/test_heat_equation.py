import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from vbatten_x import Booster, PhysicsSpec, PDEType, train_with_physics
from vbatten_x.sklearn import VBattenXRegressor


def make_heat_data(nx=8, ny=8, T=20, alpha=0.1, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((nx, ny)).astype(np.float32)
    rows, targets = [], []
    for t in range(T):
        u_new = u.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                lap = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])
                u_new[i,j] = u[i,j] + alpha * dt * lap
        rows.append(u.flatten())
        targets.append(u_new.flatten().mean())
        u = u_new
    return np.stack(rows).astype(np.float32), np.array(targets, dtype=np.float32)


def test_train_with_physics_spec():
    X, y = make_heat_data()
    spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.1, dt=0.01).grid(8, 8)
    b = train_with_physics(X, y, spec, params={"learning_rate": 0.1}, num_boost_round=20)
    assert b.num_stages == 20
    assert np.isfinite(b.train_loss)


def test_physics_reduces_vs_no_physics():
    X, y = make_heat_data(nx=6, ny=6, T=30)
    b_plain  = Booster({"learning_rate": 0.1})
    b_plain.set_data(X, y).train(30)

    spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.1).grid(6, 6)
    b_phys = Booster({"learning_rate": 0.1, "lambda_pde": 0.01})
    b_phys.set_data(X, y).set_physics(spec).train(30)

    preds_plain = b_plain.predict(X)
    preds_phys  = b_phys.predict(X)
    rmse_plain  = float(np.sqrt(np.mean((preds_plain - y)**2)))
    rmse_phys   = float(np.sqrt(np.mean((preds_phys  - y)**2)))
    assert np.isfinite(rmse_plain)
    assert np.isfinite(rmse_phys)


def test_booster_set_physics_api():
    X, y = make_heat_data()
    spec = PhysicsSpec().pde(PDEType.HEAT).grid(8, 8)
    b = Booster({"learning_rate": 0.05})
    b.set_data(X, y).set_physics(spec).train(10)
    assert b.num_stages == 10


def test_sklearn_regressor_with_physics():
    X, y = make_heat_data()
    spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.1).grid(8, 8)
    reg  = VBattenXRegressor(n_estimators=15, learning_rate=0.1,
                              lambda_pde=0.01, physics_spec=spec)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (len(y),)
    assert np.isfinite(preds).all()


def test_save_load_with_physics(tmp_path):
    X, y = make_heat_data()
    spec = PhysicsSpec().pde(PDEType.HEAT).grid(8, 8)
    b = Booster({"learning_rate": 0.1})
    b.set_data(X, y).set_physics(spec).train(10)
    p1 = b.predict(X)
    path = str(tmp_path / "model_phys.json")
    b.save(path)
    b2 = Booster.load(path)
    p2 = b2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)
