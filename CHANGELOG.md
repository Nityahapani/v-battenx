# Changelog

All notable changes to V-BATTEN-X are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v5] — Production Release

### Added
- `VBATTENX_ABI_VERSION = 5` macro — stable ABI commitment from v5 onward
- `GlobalConfig` singleton — thread count, GPU id, log level
- `Model` ABC — `Save`/`Load`/`FormatVersion`
- Complete C API: `vbx_get_metric`, `vbx_get_mutation_log`, `vbx_abi_version`, `vbx_version_string` (15 exported symbols total)
- Null-pointer guards on all C ABI entry points
- `PhysicalDataset` Python class — `from_numpy`, `from_pandas`, `from_csv`; accepted by `train()`, `cv()`, `Booster.set_data/predict`
- `Booster.get_metric(name)`, `.abi_version`, `.lib_version` properties
- `LearningRateScheduler` with `.cosine()` and `.step()` factories
- `PhysicsResidualMonitor.converged` property
- `field_viz.py` — 6 plot functions: `plot_pde_residuals`, `plot_mutation_history`, `plot_dimension_map`, `plot_field_slice`, `plot_topology`, `plot_topology_evolution`
- `compat.to_jax_array`, `compat.field_params_to_numpy`
- `training.early_stopping()` helper function
- Full R package — `vbx.train/predict/cv/save/load`, S3 print/summary/plot, `vbx.PhysicsSpec` builder, Rcpp bridge
- `doc/architecture.md` — component diagram, XGBoost/PINN/FNO comparison table
- `doc/boosting_stages.md` — variational vs classical boosting, stage model, complexity budget
- `doc/serialization.md` — format spec frozen, precision guarantee, forward compat rules
- `doc/contributing.md` — build instructions, code style, how-to add PDE/mutation, PR checklist
- Demo scripts 04–06 — custom PDE, distributed (99.5% improvement), dimension ablation
- 33 new v5 tests (91/91 total)

### Changed
- Version bumped to `5.0.0` across all files
- Model JSON format version is now `"5.0.0"`
- `train()` and `cv()` now accept `PhysicalDataset` directly
- `__init__.py` exports all callbacks and `PhysicalDataset`

---

## [v4] — Learned DTDO & Scale

### Added
- `DtdoNet` — per-region MLP + graph pooling + logit head (learned DTDO)
- `ActionSpace`, `MutationPolicy` (greedy + stochastic)
- `DtdoTrainer` — REINFORCE with advantage normalisation + imitation learning
- `AdamOptimizer`, `LbfgsOptimizer`, `ConstrainedOptimizer`, `TensorOptimizer`
- CUDA stubs — `GpuFieldState`, tensor contraction kernel, PDE ops, fused Adam (all `#ifdef VBATTENX_CUDA`)
- `Communicator` ABC, `LocalCommunicator`, `RowPartitioner`
- `DaskBooster`, `train_dask()` Dask distributed interface
- Plugin system — `VBATTENX_REGISTER_OPERATOR/PDE` macros
- Plugins: `anisotropic_expand`, Poisson PDE, `FederatedCommunicator` (signSGD + DP noise)
- 4 benchmark programs + `timer.h`; tensor contraction: 0.04–0.34 μs/call
- DTDO strategies `"learned"` and `"learned_greedy"` in operator registry
- 11 new v4 tests (63/63 total)

---

## [v3] — DTDO: Dynamic Topological-Dimensional Operator

### Added
- `MutationType` enum (10 types), `MutationLog`, `MutationEvent`
- `TopologicalOperator` ABC, `ComplexityCost` with `CanAfford/FromState/OverBudget`
- `AdaptiveDim`, `dim_transitions` (PCA expand/collapse, Frobenius-preserving, rel_err=0)
- `SimplexComplex` header — Betti0/1, graph diameter, edge density
- All 10 atomic mutations as standalone functions on `FieldState`
- `ThresholdOperator`, `GradientOperator`, `ComplexityPruner`, `MutationRouter`
- Operator registry — `"threshold"`, `"gradient"`, `"pruner"`, `"router"`, `"none"`
- `VariationalEnsemble` with `MutationLog` + PDE residuals per stage
- `TopologySnapshot/Delta`, `DimMetrics`
- `RankAdaptiveTensor` — identity-init, auto-adapts rank
- Learner v3: DTDO wired between eval and boost; v3 JSON model format
- `MutationEvent` Python class, `Booster.get_mutation_log/get_stage_pde_residuals`
- DTDO, tau, max_dim params in sklearn estimators
- 6 C++ dim transition tests, 15 new Python tests (52/52 total)
- `doc/dtdo.md` — full algorithmic reference

---

## [v2] — Physics Layer & PDE Evaluation

### Added
- Extended `PhysicalMetaInfo` — PDE params, BCs, symmetry groups, conserved quantities
- `PhysicsSpec` C++ builder + JSON serialisation
- `grid_field.h` — `GridField` + inline FD operators (no ODR issues)
- `HeatEquationEvaluator`, `NavierStokesEvaluator`, `CustomPdeEvaluator`
- Constraint checkers: symmetry, energy drift, positivity, Lagrangian penalty
- `PdeConstrainedObjective`, `PhysicsInformedObjective`
- PDE residual L2/Linf + symmetry violation metrics
- `FourierEncoder` — random Fourier features for periodic physics
- `lambda_pde` param, `vbx_set_physics` C ABI function
- Python `PhysicsSpec` fluent builder, `train_with_physics()`, `Booster.set_physics()`
- 21 new tests (41/41 total), `doc/physics_guide.md`

---

## [v1] — Foundation

### Added
- Core type system: `vbx_float`, `DeviceContext`, `Span<T>`, `HostDeviceVector<T>`
- `VBXParameter`, `JsonValue` parser/writer (full-precision float64)
- `PhysicalDataset` ABC + `DensePhysicalDataset`
- Four-tuple ABCs: `LatentField`, `FieldTopology`, `DimensionMap`, `TensorField`
- `FieldState` with `Clone()` + `Diff()`
- `ContinuousField` (Eigen), `RegionGraph`, `UniformDim`, `Rank2Tensor`
- `MlpEncoder` — 3-layer GELU MLP
- `LinearBooster` — weighted least-squares per stage
- `VBattenLearner` outer loop — train/predict/save/load
- `CpuPredictor`, `NullEvaluator`
- C ABI: 10 functions
- Python: `Booster`, `train/cv`, `VBattenXRegressor/Classifier`, callbacks, compat
- Unity build (`src/vbatten_x_impl.cc`)
- 20 tests (20/20), `demo/01_basic_regression.py`
