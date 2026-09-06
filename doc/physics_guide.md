# Physics Guide

V-BATTEN-X lets you declare physical knowledge about your system and have it shape training through the `PhysicsSpec` API. This guide covers how to define PDEs, symmetries, conservation laws, and boundary conditions, and what each does inside the training loop.

---

## PhysicsSpec

`PhysicsSpec` is a fluent builder. Every method returns `self`, so calls chain:

```python
from vbatten_x import PhysicsSpec, PDEType, SymmetryGroup, BCType

spec = (PhysicsSpec()
        .pde(PDEType.HEAT, diffusivity=0.01, dt=0.005)
        .symmetry(SymmetryGroup.ROTATION_2D)
        .conserve("energy")
        .boundary("wall", BCType.DIRICHLET, value=0.0)
        .grid(nx=32, ny=32, resolution=0.5))
```

It serialises to JSON with `spec.to_json()` and deserialises with `PhysicsSpec.from_json(s)`. The spec is passed to the learner before training:

```python
booster = Booster({"learning_rate": 0.1, "lambda_pde": 0.05})
booster.set_data(X, y).set_physics(spec).train(100)
```

Or via the convenience function:

```python
from vbatten_x import train_with_physics
b = train_with_physics(X, y, spec, params={"learning_rate": 0.1, "lambda_pde": 0.05})
```

---

## Supported PDEs

| `PDEType` | Equation | Key parameters |
|-----------|----------|----------------|
| `HEAT` | ∂u/∂t = α∇²u | `diffusivity` (α), `dt` |
| `WAVE` | ∂²u/∂t² = c²∇²u | `diffusivity` (c), `dt` |
| `NAVIER_STOKES` | Incompressible NS | `viscosity` (ν), `dt` |
| `POISSON` | ∇²u = f | — |
| `CUSTOM` | User-defined | via `MakeCustomPdeEvaluator` |

The PDE evaluator computes a per-region residual at each boosting stage. When `lambda_pde > 0`, this residual is added to the gradient signal, softly penalising field configurations that violate the declared PDE.

### Tuning `lambda_pde`

- `lambda_pde = 0.0` — pure task loss, no physics influence (default)
- `lambda_pde = 0.01` — gentle physics regularisation; use when data is dense
- `lambda_pde = 0.1`  — stronger physics bias; use when data is sparse or noisy
- `lambda_pde > 1.0`  — physics-dominant; predictions approach the PDE solution regardless of labels

### Grid configuration

PDE operators act on a spatial grid. Set its size with `.grid(nx, ny, resolution)`. The grid should match the spatial structure of your input features. For 1D problems set `ny=1`.

---

## Symmetry Groups

Declaring a symmetry tells the encoder to bias its representations toward equivariant structures.

| `SymmetryGroup` | Meaning |
|-----------------|---------|
| `ROTATION_2D`   | Field invariant under 2D rotation |
| `ROTATION_3D`   | Field invariant under 3D rotation |
| `REFLECTION`    | Field invariant under axis reflection |
| `PERMUTATION`   | Field invariant under feature permutation |
| `TRANSLATION`   | Field invariant under spatial shift |

Multiple symmetries can be declared and all are enforced simultaneously.

---

## Conservation Laws

`.conserve("energy")` and `.conserve("mass")` record the conserved quantities in the spec. The constraint checker monitors drift across boosting stages and feeds it into the Lagrangian penalty.

---

## Boundary Conditions

| `BCType`    | Enforcement |
|-------------|-------------|
| `DIRICHLET` | Hard-sets field value at boundary nodes to `value` |
| `NEUMANN`   | Sets normal derivative to `flux` at boundary nodes |
| `PERIODIC`  | Wraps field values at domain edges |

Boundary regions are named strings (e.g. `"wall"`, `"left"`, `"right"`) mapped to their condition in the spec.

---

## Custom PDEs

Implement a custom PDE residual in C++ by subclassing `PhysicsEvaluator` or using `MakeCustomPdeEvaluator` with a `std::function`. See `plugin/custom_pde/example_poisson.cc` for a reference implementation.

---

## What happens internally

Each boosting stage:

1. `PhysicsEvaluator.Eval(field_state, dataset)` computes `ResidualInfo`
2. If `lambda_pde > 0`: `g += 2 * lambda_pde * pde_residual` per sample
3. `LinearBooster.DoBoost(dataset, gp)` fits the modified gradient
4. `train_loss += lambda_pde * pde_residual²`

The `LagrangianPenalty` accumulates multipliers for conservation constraints across stages and applies dual updates every iteration.
