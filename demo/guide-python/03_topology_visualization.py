import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import json
import numpy as np
from vbatten_x import Booster, PhysicsSpec, PDEType


def make_data(n=200, d=6, seed=42):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = (np.sin(X[:, 0]) + X[:, 1] ** 2 - 0.5 * X[:, 2]).astype(np.float32)
    return X, y


X, y = make_data()

b_plain = Booster({"learning_rate": 0.1})
b_plain.set_data(X, y).train(30)

b_dtdo = Booster({
    "dtdo":         "threshold",
    "tau_expand":   0.001,
    "tau_collapse": 0.0001,
    "learning_rate": 0.1,
    "max_total_dim": 16,
})
b_dtdo.set_data(X, y).train(30)

import tempfile, os as _os
with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    path = f.name
b_dtdo.save(path)

with open(path) as f:
    model_data = json.load(f)

mutations_per_stage = [
    len(s.get("mutations", [])) for s in model_data["stages"]
]
total_mutations = sum(mutations_per_stage)

rmse = lambda pred, tgt: float(np.sqrt(np.mean((pred - tgt) ** 2)))

print("=== Topology Evolution — DTDO Demo ===\n")
print(f"Plain booster  RMSE: {rmse(b_plain.predict(X), y):.4f}")
print(f"DTDO booster   RMSE: {rmse(b_dtdo.predict(X), y):.4f}")
print(f"\nTotal topology mutations: {total_mutations}")
print(f"Stages with mutations:    {sum(1 for m in mutations_per_stage if m > 0)}/{len(mutations_per_stage)}")

if total_mutations > 0:
    print("\nMutation log (first 5 events):")
    log = b_dtdo.get_mutation_log()
    count = 0
    for stage_idx, stage_events in enumerate(log):
        for ev in stage_events:
            print(f"  stage={stage_idx:3d}  type={ev.type:<20s}  pde_r={ev.pde_r:.4f}")
            count += 1
            if count >= 5: break
        if count >= 5: break

_os.unlink(path)
