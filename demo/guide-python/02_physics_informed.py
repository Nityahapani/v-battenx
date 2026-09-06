import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
from vbatten_x import PhysicsSpec, PDEType, train, train_with_physics


def make_heat_data(nx=10, ny=10, steps=40, alpha=0.1, dt=0.01, seed=42):
    rng = np.random.default_rng(seed)
    u   = rng.standard_normal((nx, ny)).astype(np.float32)
    X, y = [], []
    for _ in range(steps):
        u_new = u.copy()
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                lap = (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4*u[i,j])
                u_new[i,j] = u[i,j] + alpha * dt * lap
        X.append(u.flatten())
        y.append(u_new.mean())
        u = u_new
    return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32)


X, y = make_heat_data()

b_plain = train(X, y, params={"learning_rate": 0.1}, num_boost_round=50)

spec    = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.1, dt=0.01).grid(10, 10)
b_phys  = train_with_physics(
    X, y, spec,
    params={"learning_rate": 0.1, "lambda_pde": 0.05},
    num_boost_round=50)

rmse = lambda pred, target: float(np.sqrt(np.mean((pred - target)**2)))

print(f"=== Heat Equation — Physics-Informed Training ===\n")
print(f"No physics:   train RMSE = {rmse(b_plain.predict(X), y):.6f}  "
      f"loss = {b_plain.train_loss:.6f}")
print(f"With physics: train RMSE = {rmse(b_phys.predict(X),  y):.6f}  "
      f"loss = {b_phys.train_loss:.6f}")
print(f"\nPhysics spec: {spec}")
print(f"Stages:       {b_phys.num_stages}")
