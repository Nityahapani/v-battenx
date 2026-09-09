# V-BATTEN-X Architecture

## The Core Idea

V-BATTEN-X is a gradient boosting library where each boosting stage operates on a **dynamic latent field** instead of a fixed-dimensional decision tree. The field can restructure itself — changing local dimensionality, splitting and merging regions, rewiring topology — between stages in response to physics residuals.

The central abstraction is the four-tuple operator:

```
Oθ : (F, K, d, T) → (F', K', d', T')
```

| Symbol | Name | What it represents |
|--------|------|--------------------|
| **F** | LatentField | Where states live — a continuous manifold |
| **K** | FieldTopology | How regions connect — an adjacency graph |
| **d** | DimensionMap | Degrees of freedom per region — local dim |
| **T** | TensorField | How components interact — rank-2 or higher |

## Component Overview

```
PhysicalDataset
      │
      ▼
PhysicsEncoder  ──►  FieldState(F, K, d, T)
                              │
                 ┌────────────▼────────────┐
                 │    Outer Loop           │
                 │                         │
                 │  1. PhysicsEvaluator   │
                 │     → ResidualInfo     │
                 │                         │
                 │  2. DTDO.Apply         │
                 │     → FieldState'      │
                 │                         │
                 │  3. Objective.GetGrads │
                 │     → (g, h)           │
                 │                         │
                 │  4. LinearBooster      │
                 │     → StageModel       │
                 │                         │
                 │  5. Ensemble.Append    │
                 └────────────────────────┘
                              │
                              ▼
                     VariationalEnsemble
                              │
                              ▼
                       FieldPredictor
```

## Comparison with Related Systems

| Property | V-BATTEN-X | XGBoost | PINN | FNO |
|----------|-----------|---------|------|-----|
| Boosting | ✅ Variational | ✅ Gradient | ❌ | ❌ |
| Dynamic topology | ✅ DTDO | ❌ | ❌ | ❌ |
| Physics constraints | ✅ PDE residuals | ❌ | ✅ | Partial |
| Adaptive dimensionality | ✅ Per-region | ❌ | ❌ | ❌ |
| Tabular data | ✅ | ✅ | Limited | Limited |
| Learned structure | ✅ Learned DTDO | ❌ | ❌ | ❌ |

## Layer Map

```
include/vbatten_x/          ← Stable public ABI (v5+)
src/field/                  ← Four-tuple implementations
src/dtdo/                   ← DTDO operators (rule-based + learned)
src/physics/                ← PDE evaluators + constraints
src/booster/                ← Variational ensemble + optimizers
src/encoder/                ← MLP + Fourier encoders
src/collective/             ← Distributed communicators
src_cuda/                   ← GPU kernels (activated by USE_CUDA=ON)
plugin/                     ← User-extensible PDE + operator plugins
python-package/vbatten_x/   ← Python bindings
R-package/                  ← R bindings (Rcpp)
```

## ABI Stability

From v5 onward, everything in `include/vbatten_x/` is a stable ABI commitment.
`VBATTENX_ABI_VERSION` is checked at runtime. Breaking changes require a major version bump.
The C ABI (`vbx_*` functions) is the stable bridge for all language bindings.
