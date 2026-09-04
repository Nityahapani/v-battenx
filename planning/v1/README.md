# V-BATTEN-X · v1 — Foundation & Core Abstractions

> **Goal:** Get the skeleton running end-to-end on a single machine with fixed topology.
> No mutation, no DTDO, no GPU. Prove the four-tuple `(F, K, d, T)` pipeline compiles,
> trains, and predicts on real tabular data. This is the "does it work at all?" milestone.

**Target:** Internal alpha. One developer can run `python train.py` on a CSV and get a number out.

---

## 1 · Build System

- [x] `CMakeLists.txt` root — options `USE_CUDA=OFF`, `USE_MPI=OFF`, `USE_OPENMP=ON`
- [x] `CMakePresets.json` — presets: `debug`, `release`, `asan`
- [x] `Makefile` thin wrapper (`make debug`, `make release`, `make test`)
- [x] `cmake/FindEigen3.cmake` — detect Eigen3 for tensor math
- [x] `cmake/VBattenXVersion.cmake` — version stamping from git tag
- [x] `cmake/Sanitizers.cmake` — AddressSanitizer toggle
- [ ] CI skeleton: build matrix (linux/mac, CPU only) — _blocked on workflow token_
- [ ] `pyproject.toml` — PEP 517 build works (`pip install -e .`)
- [ ] `setup.cfg` — package metadata, classifiers, deps pinned

## 2 · Core Type System (`include/vbatten_x/`)

- [ ] `base.h` — define `vbx_float` (float32/float64 switch), `vbx_index`, `vbx_dim_t`, CUDA guards
- [ ] `context.h` — `DeviceContext`: thread count, GPU id (ignored in v1), RNG seed, logging level
- [ ] `version.h` — `VBATTENX_VERSION_MAJOR/MINOR/PATCH` macros
- [ ] `span.h` — non-owning span (backport for C++17 compat)
- [ ] `string_view.h` — thin wrapper / alias
- [ ] `host_device_vector.h` — CPU-only in v1, GPU path stubbed with `#ifdef VBATTENX_CUDA`

## 3 · Data Layer (`src/data/`)

- [ ] `PhysicalDataset` ABC (`include/vbatten_x/data.h`)
  - [ ] `num_rows()`, `num_cols()`, `get_row(i)`, `get_col(j)` interface
  - [ ] `PhysicalMetaInfo` struct: feature names, units, PDE type tag (can be `NONE` in v1)
- [ ] `feature_map.h` — `FeatureMap`: name → column index → dtype → physical unit string
- [ ] `physical_dataset.cc` — dense in-memory impl backed by `std::vector<vbx_float>`
- [ ] `normalizer.cc` — z-score normalization; respects physical units (don't mix metres + kg)
- [ ] `adapter.h` — NumPy structured array → `PhysicalDataset` bridge (Python side only)
- [ ] **Unit tests**
  - [ ] `test_dataset.py` — construct from numpy array, check shape, check feature names
  - [ ] round-trip: normalise → denormalise → MSE < 1e-9

## 4 · The Four-Tuple Types (ABCs only, v1)

> Implementations are stubs; the ABCs define the contract every later component depends on.

- [ ] `field.h` — `LatentField` ABC: `sample()`, `embed()`, `interpolate()`, `operator+`, `operator*`
- [ ] `topology.h` — `FieldTopology` ABC: `num_regions()`, `adjacency()`, `merge()`, `split()`
- [ ] `dimension.h` — `DimensionMap` ABC: `local_dim(region_id) → int`, `dim_budget() → float`
- [ ] `tensor_field.h` — `TensorField` ABC: `rank()`, `contraction_with()`, `adapt_to_dimension()`
- [ ] `field/field_state.h` — `FieldState` struct owning one `(F, K, d, T)` instance
  - [ ] `clone()` — deep copy
  - [ ] `diff(other)` — element-wise residual norm (for convergence check)

## 5 · Encoder — Stage 1 (`src/encoder/`)

- [ ] `encoder.h` ABC — `Encode(PhysicalDataset&) → FieldState`
- [ ] `encoder/mlp_encoder.cc` — 3-layer MLP, maps raw features → fixed-dim latent vector
  - [ ] Hidden dim configurable via `encoder_param.h`
  - [ ] Activation: GELU default
  - [ ] Output: populates `FieldState.F` with a single-region continuous field
- [ ] `encoder_param.h` — struct: hidden_dim, num_layers, activation, dropout
- [ ] **Unit tests**
  - [ ] `test_training.py` — encoder output shape matches expected latent dim
  - [ ] encoder is deterministic given fixed seed

## 6 · Field Implementations — v1 (single region, fixed dim)

- [ ] `field/manifold/continuous_field.cc`
  - [ ] Backed by `Eigen::MatrixXf`
  - [ ] `sample(x)` — bilinear interpolation for 2D, linear for 1D
  - [ ] `embed(vec)` — set field values from a flat vector
- [ ] `field/topology/region_graph.cc`
  - [ ] v1: single-region graph (no adjacency needed yet)
  - [ ] `num_regions() = 1` — validates the pipeline without topology logic
- [ ] `field/dimension/uniform_dim.cc`
  - [ ] All regions get the same fixed dim (set at construction)
  - [ ] `dim_budget()` returns `dim * num_regions`
- [ ] `field/tensor/contraction_engine.cc`
  - [ ] Einstein summation for rank-2 (matrix multiply) and rank-3
  - [ ] Uses Eigen under the hood

## 7 · Physics Evaluator — v1 Stub (`src/physics/`)

- [ ] `physics_evaluator.h` ABC — `Eval(FieldState&, PhysicalDataset&) → ResidualInfo`
- [ ] `residual_info.cc` — `ResidualInfo`: per-region `pde_residual`, `prediction_residual`, `constraint_violation`
- [ ] v1 impl: **no PDE** — all residuals = 0.0 (physics-free baseline to validate boosting loop)
- [ ] `constraints/positivity_checker.cc` — enforce non-negativity where meta says `POSITIVE` unit

## 8 · Objective & Gradient (`src/objective/`)

- [ ] `objective.h` ABC — `GetGradients(pred, label) → (g, h)` (first and second derivative)
- [ ] `regression_obj.cc` — L2 loss (MSE): `g = pred - label`, `h = 1.0`
- [ ] `classification_obj.cc` — Binary logistic: `g = sigmoid(pred) - label`, `h = p*(1-p)`

## 9 · Boosting Loop — v1 Linear Booster (`src/booster/`)

> No variational mutation yet. Just classical gradient boosting on a fixed field structure.

- [ ] `field_booster.h` ABC — `DoBoost(dataset, residuals) → FieldState`, `PredictField(dataset) → FieldPrediction`
- [ ] `booster/linear/linear_booster.cc`
  - [ ] Fits a linear model on current field's latent vectors
  - [ ] One boosting stage = one linear fit on pseudo-residuals
  - [ ] Shrinkage via learning rate η
- [ ] `booster/variational/shrinkage.cc` — learning rate schedule (constant for v1)
- [ ] `booster/variational/ensemble.cc` — ordered list of stage snapshots, additive prediction

## 10 · Learner Outer Loop (`src/learner.cc`)

- [ ] `learner.h` ABC — `UpdateOneIter()`, `Train(n_iters)`, `Predict(dataset) → predictions`
- [ ] v1 sequence per iteration:
  - [ ] `PhysicsEncoder.Encode(dataset)` → initial `FieldState`
  - [ ] `PhysicsEvaluator.Eval(field_state)` → `ResidualInfo` (zeros in v1)
  - [ ] `Objective.GetGradients(pred, label)` → `(g, h)`
  - [ ] `LinearBooster.DoBoost(dataset, g, h)` → new stage
  - [ ] Append to `Ensemble`
  - [ ] Convergence check: if `|loss_t - loss_{t-1}| < tol` → stop
- [ ] CLI: `vbatten_x train --config config.json --data train.csv --output model.vbx`

## 11 · Predictor (`src/predictor/`)

- [ ] `predictor.h` ABC — `Predict(dataset, field_state) → std::vector<vbx_float>`
- [ ] `cpu_predictor.cc` — single-threaded walk of ensemble stages, accumulate predictions
- [ ] `field_eval.cc` — evaluate one `FieldState` on a batch of rows

## 12 · Serialization (`src/`, `include/vbatten_x/`)

- [ ] `model.h` ABC — `Save(path)`, `Load(path)`
- [ ] `json.h` / `json_io.h` — lightweight JSON (header-only nlohmann or custom)
- [ ] `parameter.h` — `VBXParameter`: key-value config store, JSON-serializable
- [ ] v1 format: JSON envelope + binary blob for field tensors
- [ ] `test_serialization.cc` — save → load → predict → same output

## 13 · Python Bindings — v1

- [ ] `c_api.cc` — C ABI wrapping `VBattenLearner`: `vbx_train()`, `vbx_predict()`, `vbx_save()`, `vbx_load()`
- [ ] `python-package/vbatten_x/_libvbatten.py` — `ctypes` loader
- [ ] `python-package/vbatten_x/core.py` — `PhysicalDataset`, `Booster` Python classes
- [ ] `python-package/vbatten_x/training.py` — `train(X, y, params) → Booster`
- [ ] `python-package/vbatten_x/sklearn.py` — `VBattenXRegressor`, `VBattenXClassifier` sklearn API
- [ ] **Tests**
  - [ ] `test_sklearn_api.py` — `.fit(X, y)` → `.predict(X)` → score > baseline
  - [ ] `test_dataset.py` — numpy array in, PhysicalDataset out

## 14 · Demo

- [ ] `demo/guide-python/01_basic_regression.py` — California housing, compare vs XGBoost RMSE
- [ ] `demo/notebooks/tabular_benchmark.ipynb` — vs XGBoost/LightGBM on 3 datasets

## 15 · v1 Exit Criteria (Definition of Done)

- [ ] `pip install -e .` works on Linux + macOS
- [ ] `VBattenXRegressor().fit(X, y).predict(X)` produces RMSE within 10% of XGBoost on California Housing
- [ ] `VBattenXClassifier().fit(X, y).predict_proba(X)` AUC > 0.8 on breast cancer dataset
- [ ] Save → Load round-trip: predictions identical (bit-for-bit)
- [ ] All unit tests pass (`pytest tests/python/ tests/cpp/ -v`)
- [ ] Zero memory leaks under AddressSanitizer on the learner loop
- [ ] Code review signed off by at least 1 team member
