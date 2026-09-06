# V-BATTEN-X · v2 — Physics Layer & PDE Evaluation

> **Status: ✅ COMPLETE**
> All exit criteria met. 41/41 tests passing (20 v1 regression + 21 new v2 tests).

> **Goal:** Wire in real physics. Replace the zero-residual stub from v1 with actual PDE
> operators and constraint checkers. A user can declare that their data obeys a PDE
> and have that soft-constrain the model's predictions via `lambda_pde`.

**Target:** Internal beta. Physics-aware training on heat equation and conservation law datasets.
**Depends on:** v1 ✅

---

## 1 · Physics Metadata & Spec Builder

- [x] Extend `PhysicalMetaInfo` with:
  - [x] `pde_type` — enum: `NONE`, `HEAT`, `WAVE`, `NAVIER_STOKES`, `POISSON`, `CUSTOM`
  - [x] `pde_diffusivity`, `pde_dt`, `pde_viscosity` — PDE-specific parameters
  - [x] `symmetry_groups` — `ROTATION_2D`, `ROTATION_3D`, `REFLECTION`, `PERMUTATION`, `TRANSLATION`
  - [x] `boundary_conditions` — map of region name → `{type, value, flux}`
  - [x] `conserved_quantities` — list of strings
  - [x] `grid_nx`, `grid_ny`, `spatial_resolution` — spatial grid config
- [x] `include/vbatten_x/physics_spec.h` — `PhysicsSpec` C++ builder class
- [x] `src/data/physics_meta.cc` — `ToJson` / `FromJson` serialization
- [x] `python-package/vbatten_x/physics.py` — Python `PhysicsSpec` fluent builder:
  - [x] `.pde(type, diffusivity, dt, viscosity)`
  - [x] `.symmetry(group)`
  - [x] `.conserve(quantity)`
  - [x] `.boundary(region, type, value, flux)`
  - [x] `.grid(nx, ny, resolution)`
  - [x] `.to_json()` / `PhysicsSpec.from_json(s)`

## 2 · PDE Operator Library (`src/physics/pde/`)

- [x] `grid_field.h` — `GridField` struct + all inline FD operators (no ODR issues)
- [x] `finite_diff_ops.cc` — `grad_x/y`, `laplacian`, `divergence`, `curl_2d`, `divergence_free_residual`, `curl_free_residual`, BC enforcement (`apply_dirichlet`, `apply_neumann`, `apply_periodic`)
- [x] `conservation_ops.cc` — `energy_flux_residual`
- [x] `heat_equation.cc` — `HeatEquationEvaluator`: `∂u/∂t - α∇²u` residual
- [x] `navier_stokes.cc` — `NavierStokesEvaluator`: momentum + continuity residuals
- [x] `custom_pde.cc` — `CustomPdeEvaluator`: `std::function` plug-in interface
- [x] `evaluator_registry.cc` — `MakeEvaluatorFromMeta()` factory

## 3 · Constraint Checkers (`src/physics/constraints/`)

- [x] `symmetry_checker.cc` — rotation 2D equivariance error + reflection equivariance error
- [x] `energy_checker.cc` — `EnergyChecker`: relative energy drift `|E_t - E_{t-1}| / E_0`
- [x] `positivity_checker.cc` — count negative field values + min value tracking
- [x] `lagrangian_penalty.cc` — `LagrangianPenalty`: augmented Lagrangian with dual update `λ ← λ + ρ * violation`

## 4 · Physics-Informed Objectives (`src/objective/`)

- [x] `pde_constrained_obj.cc` — `L = L_task + λ_pde * L_pde + λ_bc * L_bc`
- [x] `physics_informed_obj.cc` — PINN-style `L = L_data + λ * Σ pde_residual²`

## 5 · PDE Metrics (`src/metric/`)

- [x] `pde_metric.cc` — `PdeResidualL2Metric`, `PdeResidualLinfMetric`, `SymmetryViolationMetric`

## 6 · Encoder v2

- [x] `fourier_encoder.cc` — `FourierEncoder`: random Fourier features `[sin(ωᵀx), cos(ωᵀx)]` for periodic physics data

## 7 · Learner + C ABI updates

- [x] `learner.cc` — physics evaluator wired in; `lambda_pde` param; PDE residual logged each iteration; `SetPhysicsSpec()` method
- [x] `c_api.cc` — `vbx_set_physics(handle, spec_json)` C ABI function added
- [x] `vbatten_x_impl.cc` — unity build updated with all v2 sources

## 8 · Python API

- [x] `_libvbatten.py` — `set_physics()` ctypes binding
- [x] `core.py` — `Booster.set_physics(spec)`
- [x] `training.py` — `train_with_physics(X, y, spec, params, num_boost_round)`
- [x] `sklearn.py` — `VBattenXRegressor` + `VBattenXClassifier` gain `lambda_pde` + `physics_spec` params
- [x] `__init__.py` — exports `PhysicsSpec`, `PDEType`, `SymmetryGroup`, `BCType`
- [x] version bumped to `0.2.0`

## 9 · Tests

- [x] `tests/python/test_physics_spec.py` — 9 tests: all builder methods, no-duplicate symmetry, JSON roundtrip, chaining, repr
- [x] `tests/physics/test_heat_equation.py` — 5 tests: train_with_physics, physics vs plain, API, sklearn, save/load
- [x] `tests/physics/test_conservation.py` — 4 tests: lambda_pde=0 equals plain, conserve spec, training with conservation, finite loss
- [x] `tests/physics/test_symmetry.py` — 5 tests: rotation spec, multiple symmetries, combined spec, roundtrip, training

## 10 · Docs & Demo

- [x] `demo/guide-python/02_physics_informed.py` — heat equation: plain vs physics-informed comparison
- [x] `doc/physics_guide.md` — PDE types, lambda_pde tuning, symmetry groups, BCs, internals

## 11 · v2 Exit Criteria — Results

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| pytest suite | all pass | 41/41 | ✅ |
| PhysicsSpec JSON roundtrip | lossless | exact | ✅ |
| train_with_physics() runs | no error | passes | ✅ |
| lambda_pde=0 equals plain | identical | max diff=0 | ✅ |
| Conservation spec builds | valid JSON | passes | ✅ |
| Symmetry spec roundtrip | lossless | exact | ✅ |
| vbx_set_physics C ABI | exported | symbol present | ✅ |
| physics_guide.md written | reviewed | ✅ | ✅ |
| v1 regression suite | 20/20 | 20/20 | ✅ |
| NS evaluator — correct sign | residual > 0 on bad field | ✅ | ✅ |

## 12 · Known Carry-Forwards into v3

- `symmetry_encoder.cc` (equivariance regularisation during MLP training) — designed, deferred; rotational symmetry currently only declared in spec, not enforced during encoding
- `test_pde_ops.cc` C++ analytic verification (laplacian of sin(x)sin(y) etc.) — planned for v3 when C++ test infra is set up
- Navier-Stokes: currently residual-only; full NS solver (pressure projection, velocity coupling) is v4
- `lambda_pde` gradient term is a scalar broadcast — v3 will make it per-region once DTDO provides region structure
