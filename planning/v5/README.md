# V-BATTEN-X · v5 — Production Hardening & Public Release

> **Status: ✅ COMPLETE**
> 91/91 tests passing (63 v1-v4 regression + 28 new v5 tests).
> Released as `v5`.

> **Goal:** Turn the research system into a production-grade library.
> Stable ABI, full language bindings, complete documentation, all demos,
> comprehensive test coverage, release artifacts.

**Target:** Public v5 release — announced and documented.
**Depends on:** v4 ✅

---

## 1 · Stable ABI Commitment

- [x] `VBATTENX_ABI_VERSION = 5` macro in `version.h`
- [x] ABI version bumped on breaking changes only
- [x] `vbx_abi_version()` C function — checked at runtime
- [x] `vbx_version_string()` C function — returns `"5.0.0"`
- [x] All `vbx_*` C functions have null-pointer guards
- [x] `GlobalConfig` singleton — `SetNumThreads`, `SetGpuId`, `SetLogLevel`
- [x] `Model` ABC — `Save`/`Load`/`FormatVersion`

## 2 · Complete C API (`src/c_api.cc`)

15 exported C functions — all with null checks and thread-local error strings:

- [x] `vbx_learner_create(params_json)` → handle
- [x] `vbx_set_data(handle, X, y, nrows, ncols)` → status
- [x] `vbx_set_physics(handle, spec_json)` → status
- [x] `vbx_train(handle, n_iters)` → status
- [x] `vbx_predict(handle, X, nrows, ncols, out)` → status
- [x] `vbx_save(handle, path)` → status
- [x] `vbx_load(handle, path)` → status
- [x] `vbx_train_loss(handle)` → float
- [x] `vbx_num_stages(handle)` → int
- [x] `vbx_get_metric(handle, name)` → double
- [x] `vbx_get_mutation_log(handle, stage)` → const char* (JSON)
- [x] `vbx_abi_version()` → int
- [x] `vbx_version_string()` → const char*
- [x] `vbx_destroy(handle)`
- [x] `vbx_last_error()` → const char*

## 3 · Python Package — Production Quality

### 3a · PhysicalDataset (`core.py`)
- [x] `PhysicalDataset.from_numpy(X, y, feature_names, units)`
- [x] `PhysicalDataset.from_pandas(df, label_col, units)`
- [x] `PhysicalDataset.from_csv(path, label_col, **kwargs)`
- [x] `__repr__` — shape, first 3 feature names
- [x] Accepted by `Booster.set_data/predict`, `train()`, `cv()`

### 3b · Booster (`core.py`)
- [x] `Booster.get_metric(name)` — `"train_loss"`, `"num_stages"`
- [x] `Booster.abi_version` property → int
- [x] `Booster.lib_version` property → str

### 3c · Training (`training.py`)
- [x] `train(X, y, ...)` — accepts `PhysicalDataset` directly
- [x] `train_with_physics(X, y, spec, ...)` — accepts `PhysicalDataset`
- [x] `cv(X, y, ...)` — accepts `PhysicalDataset`
- [x] `early_stopping(rounds, metric, min_delta)` helper function

### 3d · sklearn (`sklearn.py`)
- [x] `VBattenXRegressor` + `VBattenXClassifier` unchanged from v4
- [x] `Pipeline([StandardScaler(), VBattenXRegressor()])` works
- [x] `get_params()/set_params()` for GridSearchCV compatibility

### 3e · Callbacks (`callback.py`)
- [x] `EarlyStopping(rounds, min_delta, save_best)`
- [x] `ModelCheckpoint(path, save_period)`
- [x] `PhysicsResidualMonitor(tol, stop_on_converge)` + `.converged` property
- [x] `TopologyLogger(log_dir)`
- [x] `LearningRateScheduler(fn)` + `.cosine(lr, steps)` + `.step(lr, decay, every)` factories

### 3f · Field Visualisation (`field_viz.py`)
- [x] `plot_pde_residuals(booster, ax)` — before/after per stage
- [x] `plot_mutation_history(booster, ax)` — mutation count bar + type scatter
- [x] `plot_dimension_map(booster, stage, ax)` — field params as heatmap
- [x] `plot_field_slice(booster, stage, axis, value, ax)`
- [x] `plot_topology(booster, stage, ax)` — networkx graph
- [x] `plot_topology_evolution(booster, interval, save_gif)` — animated frames
- [x] All functions return `Axes`; no `plt.show()` calls

### 3g · Interop (`compat.py`)
- [x] `to_numpy`, `from_pandas`, `to_torch_tensor`, `from_torch_tensor`
- [x] `to_jax_array` — JAX interop
- [x] `to_scipy_sparse` — topology as sparse matrix
- [x] `field_params_to_numpy(booster, stage)` — stage field params as ndarray

### 3h · `__init__.py`
- [x] Exports all callbacks, `PhysicalDataset`, `early_stopping`
- [x] Version `5.0.0`

## 4 · R Package

- [x] `R-package/R/vbatten_x.R`:
  - [x] `vbx.train(data, label, params, nrounds)` → `vbx.Booster`
  - [x] `predict.vbx.Booster(booster, newdata)` → numeric vector
  - [x] `vbx.cv(data, label, params, nfold, nrounds)` → data.frame
  - [x] `vbx.save(booster, path)`, `vbx.load(path)`
  - [x] S3: `print.vbx.Booster`, `summary.vbx.Booster`, `plot.vbx.Booster`
- [x] `R-package/R/physics.R`:
  - [x] `vbx.physics.spec()` — constructor
  - [x] `vbx.pde(spec, type, ...)`, `vbx.symmetry`, `vbx.conserve`, `vbx.boundary`, `vbx.grid`
  - [x] `vbx.spec.to_json(spec)` — JSON serialization
  - [x] `print.vbx.PhysicsSpec`
- [x] `R-package/src/vbatten_x_R.cpp` — Rcpp bridge via `XPtr<void>` to C ABI

## 5 · Documentation

- [x] `doc/architecture.md` — component diagram, comparison table (XGBoost/PINN/FNO), ABI policy
- [x] `doc/boosting_stages.md` — variational vs classical, stage model table, convergence, complexity budget
- [x] `doc/serialization.md` — format spec frozen, field tables, precision guarantee, forward compat rules, API in all 3 languages
- [x] `doc/contributing.md` — build instructions (CPU/CMake/CUDA), code style, how to add PDE evaluator (5 steps), how to add mutation type (8 steps), PR checklist, commit convention
- [x] `doc/physics_guide.md` — from v2, reviewed
- [x] `doc/dtdo.md` — from v3, reviewed
- [x] `README.md` — quickstart, features table, DTDO strategies, installation, releases table, doc index, test status badge

## 6 · Demos

- [x] `demo/guide-python/01_basic_regression.py` — synthetic regression
- [x] `demo/guide-python/02_physics_informed.py` — heat equation comparison
- [x] `demo/guide-python/03_topology_visualization.py` — DTDO mutation log
- [x] `demo/guide-python/04_custom_pde.py` — Burgers approximation + DTDO
- [x] `demo/guide-python/05_distributed.py` — Dask, 5k rows, 99.5% improvement
- [x] `demo/guide-python/06_dimension_ablation.py` — all 4 DTDO strategies comparison

## 7 · Build System

- [x] `CMakeLists.txt` — `SOVERSION=5`, install targets, GTest integration, `USE_CUDA/MPI/OPENMP` options
- [x] `CMakePresets.json` — `debug`, `release`, `asan`, `cuda` presets
- [x] `Makefile` — thin wrapper (`make debug`, `make release`, `make asan`, `make test`)
- [x] `.gitignore` — excludes build artifacts, `__pycache__`, IDEs; keeps `_vbatten_x.so`
- [x] `amalgamation/vbatten_x_all.cc` — single-file build, sources in dependency order, usage comment

## 8 · Release Artifacts

- [x] `CHANGELOG.md` — full history v1–v5
- [x] `LICENSE` — Apache 2.0
- [x] `CONTRIBUTING.md` — redirect to `doc/contributing.md`
- [x] `pyproject.toml` — version `5.0.0`, deps, optional extras
- [x] `setup.cfg` — classifiers, metadata

## 9 · Tests

- [x] `tests/python/test_v5.py` — 28 tests:
  - [x] ABI version = 5
  - [x] lib version string = `"5.0.0"`
  - [x] model JSON version = `"5.0.0"`
  - [x] `PhysicalDataset.from_numpy`, repr, accepted by Booster and train
  - [x] `Booster.get_metric("train_loss")` and `"num_stages"`
  - [x] `Booster.abi_version` and `lib_version` properties
  - [x] EarlyStopping fires, ModelCheckpoint saves, PhysicsResidualMonitor.converged
  - [x] `LearningRateScheduler.cosine` and `.step`
  - [x] `TopologyLogger` creates files
  - [x] All 6 field_viz functions run without error
  - [x] sklearn Pipeline compatible
  - [x] sklearn estimator has all required attributes
  - [x] `compat.field_params_to_numpy`, `to_numpy`
  - [x] `early_stopping()` helper
  - [x] `cv()` with physics

## 10 · v5 Exit Criteria — Results

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| pytest suite | all pass | 91/91 | ✅ |
| C API completeness | 15 functions | 15 exported | ✅ |
| Null-pointer guards | all entry points | all guarded | ✅ |
| ABI version macro | present | `VBATTENX_ABI_VERSION=5` | ✅ |
| PhysicalDataset | 3 constructors | from_numpy/pandas/csv | ✅ |
| R package | API complete | all functions | ✅ |
| field_viz | 6 plots | all return Axes | ✅ |
| LearningRateScheduler | 2 factories | cosine + step | ✅ |
| 6 demo scripts | all run | all run clean | ✅ |
| CHANGELOG | v1–v5 | complete | ✅ |
| LICENSE | Apache 2.0 | confirmed | ✅ |
| README | with quickstart | complete | ✅ |
| CMakePresets | 4 presets | debug/release/asan/cuda | ✅ |
| Amalgamation | single-file | compiles | ✅ |
| v1-v4 regression | 63/63 | 63/63 | ✅ |

## 11 · Post-v5 / v6 Ideas

- Continuous-time boosting (ODE formulation of ensemble stages)
- Attention-based field: regions attend to each other (Transformer on topology)
- Automatic symmetry discovery — learn symmetry groups from data
- Causal field: topology encodes causal graph, DTDO respects causal constraints
- Quantum-inspired tensor networks (MPS/MERA as TensorField implementations)
- Federated learning with formal DP guarantees as a core (not plugin) feature
- AutoML wrapper: automatically choose PDE type and physics spec from data
- Multi-output prediction with per-output field structure
- Streaming/online learning variant of the boosting loop
- `check_estimator` full sklearn compliance (API changes in sklearn ≥1.8)
