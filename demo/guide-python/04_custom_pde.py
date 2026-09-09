import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
from vbatten_x import Booster, PhysicsSpec, PDEType


def burgers_residual(params, ds_unused):
    n = len(params)
    if n < 3:
        return 0.0
    # Burgers: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    # Finite difference approximation on param array treated as u(x)
    nu = 0.01
    h  = 1.0 / n
    total = 0.0
    for i in range(1, n - 1):
        dudt  = 0.0
        dudx  = (params[i+1] - params[i-1]) / (2*h)
        d2udx = (params[i+1] - 2*params[i] + params[i-1]) / (h*h)
        r     = dudt + params[i] * dudx - nu * d2udx
        total += r * r
    return float(np.sqrt(total / n))


rng = np.random.default_rng(42)
n, d = 200, 16
X = rng.standard_normal((n, d)).astype(np.float32)
y = (np.sin(X[:, 0]) - 0.5 * X[:, 1]).astype(np.float32)

# plain
b_plain = Booster({"learning_rate": 0.1})
b_plain.set_data(X, y).train(30)

# custom PDE via lambda_pde (custom_pde.cc is the C++ hook for full plugin)
b_custom = Booster({"learning_rate": 0.1, "lambda_pde": 0.05, "dtdo": "threshold",
                     "tau_expand": 0.001})
spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.01).grid(4, 4)
b_custom.set_data(X, y).set_physics(spec).train(30)

rmse = lambda b: float(np.sqrt(np.mean((b.predict(X) - y)**2)))
print("=== Custom PDE Demo (Burgers approximation) ===\n")
print(f"Plain booster  RMSE: {rmse(b_plain):.4f}")
print(f"Custom PDE     RMSE: {rmse(b_custom):.4f}")
print(f"Stages (plain):      {b_plain.num_stages}")
print(f"Stages (custom):     {b_custom.num_stages}")
