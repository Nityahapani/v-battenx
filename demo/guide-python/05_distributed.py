import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
from vbatten_x.dask import train_dask, DaskBooster

rng   = np.random.default_rng(0)
N, D  = 5000, 10
X     = rng.standard_normal((N, D)).astype(np.float32)
y     = (X @ rng.standard_normal(D).astype(np.float32))

print("=== Distributed Training Demo (Dask) ===\n")

b = train_dask(
    X, y,
    params={"learning_rate": 0.1, "dtdo": "none"},
    num_boost_round=50,
    num_workers=2,
)

preds = b.predict(X)
rmse  = float(np.sqrt(np.mean((preds - y)**2)))
mean_bl = float(np.sqrt(np.mean((y.mean() - y)**2)))

print(f"Workers:         2")
print(f"Samples:         {N}")
print(f"Features:        {D}")
print(f"Stages trained:  {b.num_stages}")
print(f"Train RMSE:      {rmse:.4f}")
print(f"Mean baseline:   {mean_bl:.4f}")
print(f"Improvement:     {(mean_bl - rmse) / mean_bl * 100:.1f}%")
