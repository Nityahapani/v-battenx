# V-BATTEN-X

**Physics-informed variational boosting over dynamic latent fields.**

V-BATTEN-X is a gradient boosting library where each stage can restructure the
model's internal field — changing dimensionality, topology, and connectivity —
in response to physics residuals. It is the only boosting library with a
**Dynamic Topological-Dimensional Operator (DTDO)**.

```
Oθ : (F, K, d, T) → (F', K', d', T')
```

---

## Quickstart

```python
from vbatten_x import train, VBattenXRegressor, PhysicsSpec, PDEType

# Plain regression
booster = train(X_train, y_train, params={"learning_rate": 0.1}, num_boost_round=100)
preds   = booster.predict(X_test)

# Physics-informed regression
spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.01).grid(16, 16)
booster = train(X_train, y_train,
                params={"learning_rate": 0.1, "lambda_pde": 0.05},
                num_boost_round=100)
booster.set_physics(spec)

# sklearn API
reg = VBattenXRegressor(n_estimators=100, learning_rate=0.1, dtdo="router")
reg.fit(X_train, y_train)

# With learned DTDO
reg = VBattenXRegressor(n_estimators=100, dtdo="learned", tau_expand=0.05)
reg.fit(X_train, y_train)
```

---

## Features

| Feature | Description |
|---------|-------------|
| **DTDO** | 10 mutation types: expand/collapse dimension, split/merge regions, add/remove edges, local dim change |
| **Physics** | Heat equation, Navier-Stokes, Poisson, custom PDEs; symmetry groups; conservation laws |
| **Learned DTDO** | Neural network policy (per-region MLP + graph pooling) trained via REINFORCE |
| **Distributed** | Dask interface; federated communicator (signSGD + DP noise) |
| **GPU ready** | CUDA kernels compiled when `USE_CUDA=ON`; no source changes needed |
| **sklearn API** | `fit/predict/predict_proba/score`; `Pipeline` compatible |
| **R package** | `vbx.train/predict/cv/save/load`; `vbx.PhysicsSpec` builder |
| **Stable ABI** | `vbx_*` C functions; `VBATTENX_ABI_VERSION=5` |

---

## DTDO Strategies

```python
# Rule-based (fast, no training needed)
{"dtdo": "none"}        # disabled — v1/v2 behaviour
{"dtdo": "threshold"}   # expand if pde_r > τ, collapse if pde_r < τ
{"dtdo": "gradient"}    # split on high residual, merge on low
{"dtdo": "pruner"}      # collapse every N stages or when over budget
{"dtdo": "router"}      # priority chain: pruner → gradient → threshold

# Learned (neural policy)
{"dtdo": "learned"}         # stochastic policy
{"dtdo": "learned_greedy"}  # greedy policy
```

---

## Installation

```bash
# Build shared library (requires Eigen3, g++≥13)
g++ -std=c++17 -O2 -shared -fPIC \
  -I. -Iinclude -I/usr/include/eigen3 \
  src/vbatten_x_impl.cc src/c_api.cc \
  -o python-package/vbatten_x/_vbatten_x.so

# Install Python package
pip install -e python-package/
```

---

## Releases

| Tag | Version | Milestone |
|-----|---------|-----------|
| `v5` | 5.0.0 | Production: stable ABI, R package, full docs |
| `v4` | 4.0.0 | Learned DTDO, GPU stubs, Dask distributed |
| `v3` | 3.0.0 | DTDO — rule-based topology/dimension mutation |
| `v2` | 2.0.0 | Physics layer — PDE evaluators, constraints |
| `v1` | 1.0.0 | Foundation — linear booster, sklearn API |

---

## Documentation

| Doc | Description |
|-----|-------------|
| [Architecture](doc/architecture.md) | System design, component overview, comparison table |
| [DTDO](doc/dtdo.md) | Mutation types, strategies, tuning guide |
| [Physics Guide](doc/physics_guide.md) | PDE types, symmetries, lambda_pde tuning |
| [Boosting Stages](doc/boosting_stages.md) | Variational vs classical boosting |
| [Serialization](doc/serialization.md) | Model format spec (frozen v5+) |
| [Contributing](doc/contributing.md) | Build, style, how to add PDEs/mutations, PR checklist |
| [Planning](planning/ROADMAP.md) | Version roadmap and carry-forwards |

---

## Test Status

```
v1: 20/20   v2: 41/41   v3: 52/52   v4: 63/63   v5: 91/91
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
