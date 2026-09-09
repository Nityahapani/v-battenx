import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
from vbatten_x import Booster

rng = np.random.default_rng(7)
n, d = 300, 8
X = rng.standard_normal((n, d)).astype(np.float32)
y = (np.sin(X[:, 0]) + X[:, 1]**2 - 0.5*X[:, 2]).astype(np.float32)

configs = [
    ("no DTDO",       {"dtdo": "none"}),
    ("threshold",     {"dtdo": "threshold", "tau_expand": 0.01}),
    ("router",        {"dtdo": "router",    "tau_expand": 0.01}),
    ("learned",       {"dtdo": "learned"}),
]

print("=== Dimension Ablation: Fixed vs Adaptive DTDO ===\n")
print(f"{'Config':<20} {'Train RMSE':>12} {'Stages':>8}")
print("-" * 42)

for name, extra_params in configs:
    p = {"learning_rate": 0.1, "verbose": 0, **extra_params}
    b = Booster(p)
    b.set_data(X, y).train(40)
    preds = b.predict(X)
    rmse  = float(np.sqrt(np.mean((preds - y)**2)))
    print(f"{name:<20} {rmse:>12.4f} {b.num_stages:>8d}")
