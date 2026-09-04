# V-BATTEN-X · v2 — Physics Layer & PDE Evaluation

> **Goal:** Wire in real physics. Replace the zero-residual stub from v1 with actual PDE
> operators and constraint checkers. The model should now be able to incorporate physical
> knowledge about the system it is learning — symmetries, conservation laws, PDEs.
>
> By the end of v2 a user should be able to say _"my data obeys the heat equation"_
> and have that soft-constrain the model's predictions.

**Target:** Internal beta. Physics-aware training on heat equation and simple conservation law datasets.
**Depends on:** v1 complete and passing all exit criteria.

---

## 1 · Physics Metadata & Spec Builder

- [ ] Extend `PhysicalMetaInfo` with:
  - [ ] `pde_type` — enum: `NONE`, `HEAT`, `WAVE`, `NAVIER_STOKES`, `POISSON`, `CUSTOM`
  - [ ] `symmetry_groups` — list of: `ROTATION_2D`, `ROTATION_3D`, `REFLECTION`, `PERMUTATION`, `TRANSLATION`
  - [ ] `boundary_conditions` — map of region_id → BC type (`DIRICHLET`, `NEUMANN`, `PERIODIC`)
  - [ ] `conserved_quantities` — list of strings (e.g. `"energy"`, `"mass"`)
- [ ] `python-package/vbatten_x/physics.py` — `PhysicsSpec` builder API:
  ```python
  spec = PhysicsSpec()
      .pde(PDEType.HEAT, diffusivity=0.01)
      .symmetry(SymmetryGroup.ROTATION_2D)
      .conserve("energy")
      .boundary(BCType.DIRICHLET, value=0.0)
  ```
- [ ] `test_physics_spec.py` — spec builds, serialises to JSON, deserialises correctly

## 2 · PDE Operator Library (`src/physics/pde/`)

### 2a · Finite Difference Operators

- [ ] `finite_diff_ops.cc` — operators on `LatentField`:
  - [ ] `grad_x(field, region)` — ∂f/∂x via central differences
  - [ ] `grad_y(field, region)` — ∂f/∂y (only if local_dim ≥ 2)
  - [ ] `grad_z(field, region)` — ∂f/∂z (only if local_dim ≥ 3)
  - [ ] `laplacian(field, region)` — ∇²f in current local dimension
  - [ ] `divergence(vec_field, region)` — ∇·F
  - [ ] `curl_2d(vec_field, region)` — scalar curl for 2D fields
  - [ ] Grid spacing `h` taken from `PhysicalMetaInfo.spatial_resolution`
- [ ] **Analytic verification tests** (`test_pde_ops.cc`):
  - [ ] `laplacian(sin(x) * sin(y))` == `-2 * sin(x) * sin(y)` within tolerance `1e-4`
  - [ ] `divergence([cos(x), sin(y)])` == `-sin(x) + cos(y)` within tolerance `1e-4`
  - [ ] `grad_x(x^2)` == `2x` within tolerance `1e-4`

### 2b · Conservation Operators

- [ ] `conservation_ops.cc`:
  - [ ] `divergence_free_residual(field)` — ‖∇·F‖₂ (should be 0 for incompressible flow)
  - [ ] `curl_free_residual(field)` — ‖∇×F‖₂ (should be 0 for irrotational flow)
  - [ ] `energy_flux_residual(field, dt)` — dE/dt + ∇·J residual

### 2c · Boundary Condition Enforcement

- [ ] `boundary_ops.cc`:
  - [ ] `apply_dirichlet(field, boundary_mask, value)` — hard set at boundary nodes
  - [ ] `apply_neumann(field, boundary_mask, flux)` — normal derivative at boundary
  - [ ] `apply_periodic(field, axis)` — wrap-around for periodic domains
  - [ ] BC violation residual: `bc_residual(field, bc_spec)` → scalar

### 2d · Built-in PDE Evaluators

- [ ] `heat_equation.cc` — residual: `∂u/∂t - α∇²u`
  - [ ] Takes current field + time step info from dataset metadata
  - [ ] Returns per-region `pde_residual` scalar
  - [ ] `test_heat_equation.py` — train on synthetic heat diffusion data; check PDE residual < 0.01 after 50 iters
- [ ] `navier_stokes.cc` — incompressible NS residual:
  - [ ] Momentum: `∂u/∂t + (u·∇)u + ∇p - ν∇²u`
  - [ ] Continuity: `∇·u = 0`
  - [ ] Warning: v2 only evaluates residual — full NS solver is v4+
  - [ ] `test_navier_stokes.py` — residual evaluator gives correct sign on known-bad fields
- [ ] `custom_pde.cc` — user-defined PDE plug-in interface:
  - [ ] `CustomPDE` ABC: `Residual(FieldState&, PhysicalDataset&) → Eigen::VectorXf`
  - [ ] `plugin/custom_pde/example_poisson.cc` — Poisson equation `∇²u = f` as a reference impl

## 3 · Constraint Checkers (`src/physics/constraints/`)

- [ ] `symmetry_checker.cc`:
  - [ ] For each declared symmetry group, apply the group action to the field
  - [ ] Measure `‖f(Rx) - Rf(x)‖₂` — equivariance error
  - [ ] Returns `constraint_violation` per region
- [ ] `energy_checker.cc`:
  - [ ] Compute total energy `E = ½∫|f|² dΩ` over current field
  - [ ] Compare to previous step: `|E_t - E_{t-1}| / E_0` — relative energy drift
  - [ ] Flag if drift > configurable threshold
- [ ] `positivity_checker.cc`:
  - [ ] For features marked `POSITIVE` in `PhysicalMetaInfo` (e.g. density, temperature)
  - [ ] Count regions where field values go negative → hard constraint violation
- [ ] `lagrangian_penalty.cc`:
  - [ ] Maintains Lagrange multipliers λ per constraint per region
  - [ ] `penalty(field, lambdas) → scalar` — soft constraint as additive loss term
  - [ ] Dual update: `λ ← λ + ρ * constraint_violation` (augmented Lagrangian)
  - [ ] `test_conservation.py` — energy conservation enforced within 1% over 100 boosting stages
  - [ ] `test_symmetry.py` — equivariance error < 0.05 when rotation symmetry declared

## 4 · Physics-Informed Objective (`src/objective/`)

- [ ] `pde_constrained_obj.cc`:
  - [ ] Combined loss: `L = L_task + λ_pde * L_pde + λ_bc * L_bc`
  - [ ] Gradient `g` includes PDE adjoint term (finite-difference adjoint for v2)
  - [ ] Hessian diagonal `h` approximated (ignore cross terms for now)
  - [ ] λ values configurable in `parameter.h`
- [ ] `physics_informed_obj.cc` — PINN-style:
  - [ ] `L = L_data + λ_physics * Σ_regions(pde_residual²)`
  - [ ] Works without labelled data (pure PDE fitting mode)
  - [ ] Requires `PhysicsSpec` to be set

## 5 · PDE Metrics (`src/metric/`)

- [ ] `pde_metric.cc`:
  - [ ] `PDE_RESIDUAL_L2` — L2 norm of PDE residual across all regions
  - [ ] `PDE_RESIDUAL_LINF` — max PDE residual (worst region)
  - [ ] Both reported per boosting stage in training log
- [ ] `symmetry_metric.cc`:
  - [ ] `SYMMETRY_VIOLATION` — mean equivariance error across all declared symmetry groups
- [ ] Update `learner.cc` to log these metrics each iteration alongside RMSE

## 6 · Encoder v2 — Symmetry-Aware (`src/encoder/`)

- [ ] `symmetry_encoder.cc`:
  - [ ] Wraps `MlpEncoder` with equivariance layer
  - [ ] For rotation symmetry: applies random rotations to input, checks output rotates accordingly
  - [ ] Training: equivariance regularisation term added to encoder loss
  - [ ] Only active when `PhysicsSpec` declares a symmetry group
- [ ] `fourier_encoder.cc`:
  - [ ] Spectral embedding for data with known periodicity (e.g. time series, wave phenomena)
  - [ ] `embed(x) = [sin(ω₁x), cos(ω₁x), ..., sin(ωₙx), cos(ωₙx)]` + learned weights
  - [ ] Frequencies ω initialised from data via FFT analysis of training set

## 7 · Python Physics API

- [ ] `python-package/vbatten_x/physics.py` — full `PhysicsSpec` class:
  - [ ] `.pde(type, **kwargs)` — declare PDE and its parameters
  - [ ] `.symmetry(group)` — declare symmetry group
  - [ ] `.conserve(quantity)` — declare conserved quantity
  - [ ] `.boundary(type, **kwargs)` — declare boundary conditions
  - [ ] `.to_json()` / `PhysicsSpec.from_json(s)` — serialization
- [ ] `training.py` — `train_with_physics(X, y, physics_spec, params)`
- [ ] **Tests**:
  - [ ] `test_physics_spec.py` — all builder methods, JSON round-trip
  - [ ] `test_training.py` — physics-informed training reduces PDE residual vs no-physics baseline

## 8 · Demo

- [ ] `demo/guide-python/02_physics_informed.py`:
  - [ ] Synthetic 2D heat diffusion dataset
  - [ ] Train with and without `PhysicsSpec(PDEType.HEAT)`
  - [ ] Plot: PDE residual over boosting stages, comparison
- [ ] `demo/notebooks/heat_equation_tutorial.ipynb` — step-by-step walkthrough

## 9 · v2 Exit Criteria (Definition of Done)

- [ ] `PhysicsSpec` round-trips through JSON without data loss
- [ ] Heat equation PDE residual L2 < 0.01 after 50 boosting stages on synthetic data
- [ ] Energy conservation error < 1% when `conserve("energy")` is declared
- [ ] Equivariance error < 0.05 when `symmetry(ROTATION_2D)` is declared
- [ ] PDE residual metrics appear in training log every iteration
- [ ] `test_pde_ops.cc` analytic verification all pass
- [ ] `02_physics_informed.py` demo runs in < 60 seconds on CPU
- [ ] No regression on v1 exit criteria (run v1 test suite before merging)
- [ ] Physics guide doc (`doc/physics_guide.md`) written and reviewed
