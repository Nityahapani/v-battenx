# V-BATTEN-X · v5 — Production Hardening, Full API & Public Release

> **Goal:** Turn the research system into a production-grade library.
> Stable ABI, full language bindings, complete documentation, performance
> parity with XGBoost on tabular benchmarks, and published results on physics
> benchmarks. This is the version that gets announced.
>
> Everything in v5 is about _polish, stability, and reach_ — not new algorithmic ideas.
> New algorithmic ideas go into a `v6` design doc once v5 ships.

**Target:** Public v1.0.0 release on PyPI, CRAN, Maven Central.
**Depends on:** v4 complete and stable on at least one GPU cluster.

---

## 1 · Stable ABI Commitment

> Everything in `include/vbatten_x/` becomes a **stable ABI promise**.
> Breaking changes require a major version bump. This section locks down the contract.

- [ ] Audit every header in `include/vbatten_x/`:
  - [ ] Remove any `std::` types from public interfaces (ABI-breaking across compilers)
  - [ ] Replace `std::vector` return types with `span<T>` or explicit `size + pointer`
  - [ ] Replace `std::string` parameters with `string_view`
  - [ ] All `virtual` destructors present
  - [ ] No exceptions across ABI boundary — use `Result<T, ErrorCode>` pattern
- [ ] `version.h` — `VBATTENX_ABI_VERSION` macro; checked at load time in `c_api.cc`
- [ ] ABI stability test suite:
  - [ ] Compile a "v5 ABI consumer" shared library against v5 headers
  - [ ] Link it against v5 + v5.1 + v5.2 shared libs — must not segfault
  - [ ] `test_c_api.cc` — all C ABI functions callable with correct outputs
- [ ] `doc/serialization.md` — model file format frozen; format version in file header

## 2 · C API Completeness (`src/c_api.cc`)

- [ ] Full coverage of `VBattenLearner` via C functions:
  - [ ] `vbx_learner_create(config_json) → VBXHandle`
  - [ ] `vbx_learner_set_data(handle, X, y, nrows, ncols) → VBXStatus`
  - [ ] `vbx_learner_set_physics(handle, physics_json) → VBXStatus`
  - [ ] `vbx_learner_train(handle, n_iters) → VBXStatus`
  - [ ] `vbx_learner_predict(handle, X, nrows, out_preds) → VBXStatus`
  - [ ] `vbx_learner_save(handle, path) → VBXStatus`
  - [ ] `vbx_learner_load(path) → VBXHandle`
  - [ ] `vbx_learner_destroy(handle)`
  - [ ] `vbx_get_metric(handle, name) → double`
  - [ ] `vbx_get_mutation_log(handle, stage) → const char* (JSON)`
  - [ ] `vbx_last_error() → const char*` — thread-local error string
- [ ] All functions: null-pointer checks, error string set on failure
- [ ] `test_c_api.cc` — every function exercised; error paths tested

## 3 · Python Package — Production Quality

### 3a · Core API (`python-package/vbatten_x/`)

- [ ] `core.py` — `PhysicalDataset`:
  - [ ] `from_numpy(X, feature_names=None, units=None)` — primary constructor
  - [ ] `from_pandas(df, units=None)` — pandas DataFrame support
  - [ ] `from_arrow(table)` — Apache Arrow support
  - [ ] `from_csv(path, **kwargs)` — convenience loader
  - [ ] `__repr__` — informative string: shape, feature names, units
- [ ] `core.py` — `Booster`:
  - [ ] `train(dataset, params, num_boost_round, evals, callbacks) → Booster`
  - [ ] `predict(dataset) → np.ndarray`
  - [ ] `save_model(path)`, `load_model(path)` — classmethod
  - [ ] `get_field_state(stage) → FieldStateProxy` — inspect latent field
  - [ ] `get_mutation_log() → List[MutationEvent]`
  - [ ] `feature_importances_` — property: contribution of each feature across stages
- [ ] `training.py`:
  - [ ] `train(params, dtrain, num_boost_round, evals, callbacks, verbose_eval) → Booster`
  - [ ] `cv(params, data, nfold, num_boost_round) → CVResult` — cross-validation
  - [ ] `early_stopping(rounds, metric, min_delta)` — helper

### 3b · sklearn API (`sklearn.py`)

- [ ] `VBattenXRegressor(BaseEstimator, RegressorMixin)`:
  - [ ] `__init__(n_estimators, learning_rate, max_dim, pde_type, **params)`
  - [ ] `fit(X, y, sample_weight=None, eval_set=None)`
  - [ ] `predict(X) → np.ndarray`
  - [ ] `score(X, y) → float` (R²)
  - [ ] `feature_importances_` property
  - [ ] `get_params()` / `set_params()` — for GridSearchCV compatibility
- [ ] `VBattenXClassifier(BaseEstimator, ClassifierMixin)`:
  - [ ] All of the above + `predict_proba(X) → np.ndarray`
  - [ ] Multi-class support via softmax objective
- [ ] `test_sklearn_api.py`:
  - [ ] `GridSearchCV` works on `VBattenXRegressor`
  - [ ] `Pipeline([scaler, VBattenXClassifier()])` works
  - [ ] `check_estimator(VBattenXRegressor())` passes sklearn's internal checks

### 3c · Callbacks (`callback.py`)

- [ ] `EarlyStopping(rounds, metric, save_best=True)`
- [ ] `ModelCheckpoint(path, save_period=10)`
- [ ] `PhysicsResidualMonitor(tol, stop_on_converge=False)` — stops if PDE residual < tol
- [ ] `TopologyLogger(log_dir)` — saves `MutationLog` JSON each stage
- [ ] `LearningRateScheduler(schedule_fn)` — custom LR schedule

### 3d · Field Visualisation (`field_viz.py`) — Production Quality

- [ ] `plot_topology(booster, stage=-1, ax=None)` — networkx + matplotlib
- [ ] `plot_topology_evolution(booster, interval=1, save_gif=None)` — animated
- [ ] `plot_dimension_map(booster, stage=-1)` — heatmap of local_dim per region
- [ ] `plot_pde_residuals(booster)` — residual vs stage for each region
- [ ] `plot_mutation_history(booster)` — Gantt-style chart of mutations over stages
- [ ] `plot_field_slice(booster, stage, axis=0, value=0.0)` — 2D slice of latent field
- [ ] All plots: matplotlib-compatible, return `Axes`; no `plt.show()` calls
- [ ] `test_field_viz.py` — all functions run without error; outputs are `Axes` objects

### 3e · Interop (`compat.py`)

- [ ] `to_torch_tensor(field_state) → torch.Tensor` — export field for PyTorch downstream
- [ ] `from_torch_tensor(t) → FieldState` — import from PyTorch
- [ ] `to_jax_array(field_state) → jnp.ndarray`
- [ ] `to_scipy_sparse(topology) → scipy.sparse.csr_matrix` — topology as sparse matrix

## 4 · R Package — Complete

- [ ] `R-package/R/vbatten_x.R`:
  - [ ] `vbx.train(data, params, nrounds)` → `vbx.Booster`
  - [ ] `predict(booster, newdata)` → numeric vector
  - [ ] `vbx.cv(data, params, nfold, nrounds)` → data.frame of metrics
  - [ ] `vbx.save(booster, path)`, `vbx.load(path)`
  - [ ] S3 class `vbx.Booster` with `print`, `summary`, `plot` methods
- [ ] `R-package/R/physics.R`:
  - [ ] `vbx.physics.spec()` — builder
  - [ ] `.pde(type, ...)`, `.symmetry(group)`, `.conserve(quantity)`, `.boundary(...)`
- [ ] `R-package/src/vbatten_x_R.cpp` — Rcpp bridge to C API
- [ ] CRAN-compatible: `R CMD check` produces 0 errors, 0 warnings

## 5 · JVM Package (`jvm-packages/vbatten4j/`)

- [ ] Java API:
  - [ ] `VBattenXBooster.train(DMatrix, Map<String,Object> params, int rounds)`
  - [ ] `booster.predict(DMatrix) → float[][]`
  - [ ] `booster.saveModel(String path)`, `VBattenXBooster.loadModel(String path)`
- [ ] JNI bridge to C API
- [ ] Maven artifact: `io.vbattenx:vbatten4j:1.0.0`
- [ ] Scala convenience wrappers in `vbatten4j-scala`

## 6 · Documentation — Complete & Reviewed

- [ ] `doc/architecture.md` — final version:
  - [ ] Full (F,K,d,T) abstraction explanation with diagrams
  - [ ] DTDO algorithm pseudocode (both rule-based and learned)
  - [ ] Comparison table: V-BATTEN-X vs XGBoost vs PINNs vs FNO
- [ ] `doc/dtdo.md` — DTDO algorithm deep-dive:
  - [ ] Mutation type descriptions with before/after field diagrams
  - [ ] Rule-based DTDO: pseudocode + worked example
  - [ ] Learned DTDO: network architecture diagram + training procedure
  - [ ] Complexity budget: how to set it, what happens when exceeded
- [ ] `doc/physics_guide.md`:
  - [ ] How to define PDE (built-in vs custom plugin)
  - [ ] How to declare symmetries and conservation laws
  - [ ] Examples: heat equation, Navier-Stokes, Poisson, custom
  - [ ] How to tune λ_pde vs λ_task tradeoff
- [ ] `doc/boosting_stages.md`:
  - [ ] V-BATTEN-X boosting vs classical gradient boosting (diagram)
  - [ ] Why each stage can mutate the field (variational interpretation)
  - [ ] Shrinkage, ensemble weights, convergence theory
- [ ] `doc/serialization.md`:
  - [ ] Model file format spec (frozen for v5.x)
  - [ ] JSON envelope schema
  - [ ] Binary blob layout for tensor data
  - [ ] Forward compatibility rules
- [ ] `doc/contributing.md`:
  - [ ] Build instructions (Linux, macOS, Windows)
  - [ ] Code style (clang-format, pylint configs)
  - [ ] How to add a new PDE evaluator (step-by-step)
  - [ ] How to add a new mutation type
  - [ ] PR checklist: tests, docs, benchmark, no ABI break
- [ ] Auto-generated Doxygen API docs in `doc/api/`
- [ ] All docs spell-checked; at least 2 team members reviewed each doc

## 7 · Demos & Tutorials — Complete

- [ ] `demo/guide-python/01_basic_regression.py` — polished, comments in plain English
- [ ] `demo/guide-python/02_physics_informed.py` — heat equation, side-by-side comparison
- [ ] `demo/guide-python/03_topology_visualization.py` — topology GIF, annotated
- [ ] `demo/guide-python/04_custom_pde.py` — user plugs in Burgers' equation
- [ ] `demo/guide-python/05_distributed.py` — Dask cluster, 10M rows
- [ ] `demo/guide-python/06_dimension_ablation.py` — fixed dim vs adaptive dim ablation
- [ ] `demo/notebooks/heat_equation_tutorial.ipynb`:
  - [ ] Fully executed, all cells have output
  - [ ] Results reproducible with `random_seed=42`
- [ ] `demo/notebooks/navier_stokes_tutorial.ipynb`:
  - [ ] Residual evaluation on 2D lid-driven cavity problem
  - [ ] Comparison: V-BATTEN-X vs vanilla PINN on same dataset
- [ ] `demo/notebooks/tabular_benchmark.ipynb`:
  - [ ] 10 OpenML datasets
  - [ ] Competitors: XGBoost 2.0, LightGBM 4.0, CatBoost
  - [ ] Metrics: RMSE (regression), AUC (classification)
  - [ ] V-BATTEN-X target: within 5% of best competitor on all datasets

## 8 · Testing — Full Coverage

- [ ] C++ unit tests: ≥ 90% line coverage measured by gcov
- [ ] Python tests: ≥ 90% line coverage measured by coverage.py
- [ ] All physics correctness tests pass with tightened tolerances (v5 tolerances 2× tighter than v3)
- [ ] Regression test suite: 20 datasets, predictions pinned to known values (bit-reproducible)
- [ ] Memory: no leaks under Valgrind on full learner loop (Linux)
- [ ] Thread safety: learner + predictor safe under concurrent calls (`std::mutex` audit)
- [ ] Windows build: compiles cleanly under MSVC 2022 (no CUDA on Windows for v5)

## 9 · Performance Targets

- [ ] CPU (single thread): 100 boosting stages, 10k rows, 20 features, dim=2 → < 10 seconds
- [ ] CPU (8 threads, OpenMP): same as above → < 2 seconds
- [ ] GPU (A100): same as above → < 0.5 seconds
- [ ] Prediction throughput: ≥ 1M rows/second on CPU, ≥ 10M rows/second on GPU
- [ ] Memory: 1M rows, 100 features, 500 stages → < 8 GB RAM

## 10 · Release Checklist

- [ ] `CHANGELOG.md` — full history from v1 through v5
- [ ] `LICENSE` — Apache 2.0 confirmed by legal
- [ ] All `TODO` and `FIXME` comments resolved or tracked as issues
- [ ] `pyproject.toml` metadata complete: description, classifiers, homepage, keywords
- [ ] PyPI upload: `pip install vbatten-x` works from clean virtualenv
- [ ] CRAN submission: `R CMD check --as-cran` clean
- [ ] Maven artifact uploaded to Maven Central
- [ ] GitHub release: tag `v1.0.0`, release notes, pre-built wheels for linux/mac/win × py3.10/3.11/3.12
- [ ] Amalgamation build: `amalgamation/vbatten_x_all.cc` compiles as single file
- [ ] Security audit: no `system()` calls, no format string vulnerabilities, no unbounded stack allocs
- [ ] DOI registered on Zenodo for academic citation

## 11 · Post-v5 / v6 Ideas (Parking Lot)

> Not for v5. Captured here so ideas aren't lost.

- [ ] Continuous-time boosting (ODE formulation of ensemble stages)
- [ ] Attention-based field: regions attend to each other (Transformer on topology)
- [ ] Automatic symmetry discovery (learn symmetry groups from data)
- [ ] Causal field: topology encodes causal graph, DTDO respects causal constraints
- [ ] Quantum-inspired tensor networks (MPS/MERA as TensorField implementations)
- [ ] Federated learning with differential privacy (v4 plugin → core feature)
- [ ] AutoML wrapper: automatically choose PDE type and physics spec from data
