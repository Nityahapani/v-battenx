# V-BATTEN-X · v4 — Learned DTDO, GPU Acceleration & Distributed Training

> **Goal:** Replace the heuristic rule-based DTDO with a trained neural network policy
> that _learns_ when and where to mutate. Simultaneously, unlock GPU acceleration for
> the computationally heavy parts (tensor contraction, PDE ops, field eval) and add
> multi-machine distributed training via MPI/Dask.
>
> This is the version that turns V-BATTEN-X from a research prototype into a system
> that can handle real-world large-scale physics problems.

**Target:** Public beta. Benchmarks vs XGBoost/LightGBM written and publishable.
**Depends on:** v3 complete. Rule-based DTDO must be stable before training its replacement.

---

## 1 · Learned DTDO — Neural Network Policy (`src/dtdo/learned/`)

> The learned DTDO is a small neural network trained **jointly** with the main model.
> It takes the current field state + residuals and outputs a probability distribution
> over mutation types and target regions.

### 1a · Network Architecture (`dtdo_net.cc`)

- [ ] Input features per region `r`:
  - [ ] `pde_residual(r)` — scalar
  - [ ] `prediction_residual(r)` — scalar
  - [ ] `local_dim(r)` — int cast to float
  - [ ] `dim_budget_remaining` — scalar (global)
  - [ ] `num_neighbours(r)` — degree in topology graph
  - [ ] `betti_0, betti_1` — global topology features
  - [ ] Field embedding: mean-pooled tensor in region `r` → fixed-dim vector
- [ ] Architecture:
  - [ ] Per-region MLP: `(input_dim → 64 → 64)` with LayerNorm + GELU
  - [ ] Graph attention layer: aggregate neighbour features with learned attention
  - [ ] Global pooling: concat [max, mean] of all region embeddings
  - [ ] Head MLP: `(global_dim → 128 → |MutationType| × |regions|)` — logit per action
- [ ] Output:
  - [ ] `mutation_logits[mutation_type][region_id]` — raw scores
  - [ ] `no_op_logit` — probability of doing nothing this stage
  - [ ] Softmax over all `|MutationType| × |regions| + 1` options
- [ ] Implementation: pure Eigen + custom autograd (no PyTorch dependency in C++)
  - [ ] OR: expose as a Python-side policy using `torch.nn.Module` called via C API
  - [ ] Decision needed: **Discuss with team — native C++ vs Python policy** ⬅ open question

### 1b · Action Space (`action_space.cc`)

- [ ] Discrete actions: one per `(MutationType, region_id)` pair
- [ ] Continuous parameters per action:
  - [ ] For `SPLIT_REGION`: split axis direction (unit vector in `local_dim(r)` dimensions)
  - [ ] For `EXPAND_*`: new basis direction (initialised from residual PCA, fine-tuned)
  - [ ] For `LOCAL_DIM_CHANGE`: target dimension `d'`
- [ ] `ActionSpace.sample(logits, temperature)` — stochastic sampling with temperature
- [ ] `ActionSpace.greedy(logits)` — argmax (used at inference)
- [ ] Mask invalid actions: `can_afford(mutation)` must be true

### 1c · Mutation Policy (`mutation_policy.cc`)

- [ ] `GreedyPolicy` — always pick highest-logit valid action
- [ ] `StochasticPolicy` — sample from softmax with temperature τ (τ→0 = greedy)
- [ ] `BeamSearchPolicy` — maintain top-K candidate field states across K steps
  - [ ] Memory intensive; only use for offline analysis / ablation
  - [ ] Not used in production training loop
- [ ] Policy is swappable at config time: `dtdo.policy = "greedy" | "stochastic" | "beam"`

### 1d · DTDO Trainer (`dtdo_trainer.cc`)

> The DTDO is trained via a **meta-learning** signal: if a mutation at stage `t`
> leads to lower final loss than the alternative, it was a good mutation.

- [ ] Training signal:
  - [ ] For each boosting stage: record mutation taken + subsequent loss trajectory
  - [ ] If `loss_{t+5} < loss_counterfactual_{t+5}` → mutation was good → positive reward
  - [ ] REINFORCE-style gradient on `mutation_logits`
- [ ] DTDO update frequency: every `N` main boosting stages (default `N=10`)
- [ ] Experience replay buffer: store `(state, action, reward)` tuples
- [ ] DTDO learning rate separate from main model learning rate
- [ ] Option to **pre-train DTDO** on rule-based DTDO trajectories (imitation learning):
  - [ ] Collect `(state, rule_based_action)` pairs from v3 rule-based DTDO
  - [ ] Supervised train DTDO network to mimic rule-based decisions
  - [ ] Fine-tune with RL signal
- [ ] **Warning**: this is the trickiest component. Allocate 2-3x more time than estimated.

## 2 · GPU Acceleration (`src_cuda/`)

### 2a · GPU Field State (`src_cuda/field/gpu_field_state.cu`)

- [ ] `GpuFieldState` — mirrors `FieldState` but all tensors in CUDA device memory
  - [ ] `ToDevice(FieldState&) → GpuFieldState` — host → device transfer
  - [ ] `ToHost(GpuFieldState&) → FieldState` — device → host transfer
  - [ ] Pinned host memory for async transfers
  - [ ] Handle ragged tensor layout (different rank per region) in GPU memory

### 2b · Tensor Contraction Kernels (`src_cuda/field/tensor_contraction.cu`)

- [ ] Batched Einstein summation via cuBLAS `cublasSgemmBatched`
- [ ] Custom CUDA kernel for rank-3 contractions not expressible as GEMM
- [ ] `contract_all_regions(gpu_state)` — parallel contraction across all regions simultaneously
- [ ] Benchmark target: ≥ 10× speedup vs CPU on `num_regions=100`, `local_dim=3`

### 2c · GPU PDE Operators (`src_cuda/physics/gpu_pde_ops.cu`)

- [ ] `gpu_laplacian(field, h)` — parallel finite difference across grid points
- [ ] `gpu_divergence(vec_field)` — parallel divergence
- [ ] `gpu_pde_residual_all_regions(state, pde_type)` — compute residuals for all regions in one kernel launch
- [ ] Benchmark target: ≥ 5× speedup vs CPU on `grid_size=128×128`

### 2d · GPU DTDO Mutation Evaluation (`src_cuda/dtdo/gpu_mutation_eval.cu`)

- [ ] Parallel residual scanning: for each `(mutation_type, region_id)` pair, estimate expected residual reduction
- [ ] Used by learned DTDO to score candidate mutations without running them all
- [ ] Output: `mutation_scores[mutation_type][region_id]` — GPU tensor

### 2e · GPU Optimizer (`src_cuda/booster/gpu_optimizer.cu`)

- [ ] `GpuAdam` — Adam optimiser kernel (fused parameter update)
- [ ] `GpuLBFGS` — L-BFGS two-loop recursion on GPU (for constrained physics problems)
- [ ] Works on `GpuFieldState` parameters in-place

### 2f · GPU Predictor (`src/predictor/gpu_predictor.cu`)

- [ ] `GpuPredictor` — batch prediction using `GpuFieldState`
- [ ] Pipeline: `host batch → device → field eval → device → host result`
- [ ] `num_streams` configurable for overlapping compute and transfer
- [ ] `test_gpu_predictor.py` — CPU vs GPU predictions match within `1e-5`
- [ ] `test_gpu_pde_ops.py` — GPU PDE residuals match CPU within `1e-4`

### 2g · CUDA Utils (`src/common/gpu/`)

- [ ] `cuda_utils.h` — `VBATTENX_CUDA_CHECK(expr)` macro, stream management
- [ ] `memory_pool.cu` — memory pool to avoid repeated `cudaMalloc` in inner loop
  - [ ] Pre-allocate pool at startup based on `max_regions * max_local_dim * batch_size`
  - [ ] `pool.alloc(bytes) → void*`, `pool.free(ptr)`

## 3 · Distributed Training (`src/`, `include/vbatten_x/collective/`)

### 3a · Collective Communication ABCs

- [ ] `communicator.h`:
  - [ ] `Allreduce(tensor, op)` — sum/mean gradients across workers
  - [ ] `Broadcast(tensor, root)` — sync parameters from root to all
  - [ ] `Barrier()` — synchronisation point
  - [ ] `rank()` → int, `world_size()` → int
- [ ] `result.h` — `CollectiveResult`: success/failure + error message
- [ ] `socket.h` — raw socket communicator for testing (no MPI needed)

### 3b · MPI Communicator

- [ ] `src/collective/mpi_communicator.cc`:
  - [ ] Wraps `MPI_Allreduce`, `MPI_Bcast` with `vbx_float` type handling
  - [ ] Handles both float32 and float64 gracefully
  - [ ] Compiled only when `USE_MPI=ON`

### 3c · Data Partitioner (`src/data/partitioner.cc`)

- [ ] `RowPartitioner` — splits dataset rows across `world_size` workers
- [ ] Worker `i` gets rows `[i * n/W, (i+1) * n/W)`
- [ ] Ensures each worker sees a different slice each epoch (shuffle seed synced)

### 3d · Distributed Learner Loop

- [ ] Each worker runs the same learner loop on its data shard
- [ ] After `GetGradients`: `Allreduce(g, SUM)` and `Allreduce(h, SUM)` across workers
- [ ] After `DoBoost`: `Broadcast(new_stage_params, root=0)` to sync new stage
- [ ] DTDO mutations: only root (rank 0) decides; mutation applied then broadcast
  - [ ] Ensures topology is identical across all workers
- [ ] `field_viz.py` — only rank 0 logs and visualises

### 3e · Dask Interface (`python-package/vbatten_x/dask.py`)

- [ ] `DaskBooster` — wraps `Booster` for Dask distributed cluster
- [ ] `train_dask(dask_array_X, dask_array_y, params)` — distributed training
- [ ] Uses Dask futures to sync gradients across scheduler/workers
- [ ] `demo/guide-python/05_distributed.py` — example on synthetic 10M row dataset

## 4 · Optimisation Layer — v4 Upgrade (`src/booster/optimization/`)

- [ ] `field_optimizer.cc` — Adam on field parameters:
  - [ ] `m_t` and `v_t` moment estimates stored per parameter group
  - [ ] GPU path delegates to `GpuAdam`
  - [ ] `grad_clip` — gradient clipping by global norm
- [ ] `tensor_optimizer.cc` — rank-aware:
  - [ ] Treats each region's tensor independently (different ranks = different param groups)
  - [ ] Supports **rank regularisation**: adds `λ * ‖T_r‖_*` (nuclear norm penalty) to loss
- [ ] `constrained_opt.cc` — augmented Lagrangian for physics constraints:
  - [ ] Outer loop: update Lagrange multipliers
  - [ ] Inner loop: Adam on field parameters with fixed multipliers
  - [ ] Convergence: `‖constraint_violations‖ < tol`

## 5 · Plugin System (`plugin/`)

- [ ] `plugin/custom_operator/custom_operator_interface.h`:
  - [ ] User subclasses `TopologicalOperator` ABC
  - [ ] Compiled as shared library: `cmake -DVBATTENX_PLUGIN=my_op.so`
  - [ ] Loaded at runtime via `dlopen` / `LoadLibrary`
  - [ ] `example_anisotropic_expand.cc` — expands only along direction of max PDE residual gradient
- [ ] `plugin/federated/federated_communicator.cc`:
  - [ ] Communicator that never sends raw gradients (privacy-preserving)
  - [ ] Sends `sign(g)` only (like signSGD) — differential privacy sketch
  - [ ] Configurable noise `σ` for formal (ε, δ)-DP guarantee

## 6 · Benchmarks (`benchmark/`)

- [ ] `bench_boosting_loop.cc`:
  - [ ] 100-stage boosting loop, `num_regions=10`, `local_dim=2`
  - [ ] CPU vs GPU timing
  - [ ] Target: < 1ms per stage on GPU
- [ ] `bench_dtdo.cc`:
  - [ ] DTDO apply time vs number of regions (1, 5, 10, 50, 100)
  - [ ] Rule-based vs learned DTDO comparison
- [ ] `bench_tensor_contraction.cc`:
  - [ ] Rank-2, 3, 4 contraction; batch sizes 1, 32, 256
  - [ ] CPU (Eigen) vs CUDA comparison
- [ ] `bench_pde_ops.cc`:
  - [ ] Laplacian on `64²`, `128²`, `256²`, `512²` grids
  - [ ] CPU vs GPU; expect super-linear speedup for larger grids
- [ ] All benchmarks output CSV: parseable by `demo/notebooks/tabular_benchmark.ipynb`

## 7 · `cmake/FindCUDA.cmake` & Build Integration

- [ ] `CMakeLists.txt` `USE_CUDA=ON` path:
  - [ ] Detects CUDA toolkit via `find_package(CUDA)`
  - [ ] Compiles `src_cuda/` with `nvcc`; links `libcublas`, `libcudart`
  - [ ] Sets `VBATTENX_CUDA` preprocessor define
- [ ] CI: add GPU runner (A100 or T4) when workflow token is available

## 8 · v4 Exit Criteria (Definition of Done)

- [ ] Learned DTDO trains without NaN loss for 500 boosting stages
- [ ] Learned DTDO achieves ≥ 15% lower PDE residual than rule-based DTDO on heat equation
- [ ] GPU tensor contraction ≥ 10× faster than CPU on 100 regions, dim=3
- [ ] GPU PDE operators ≥ 5× faster than CPU on 128×128 grid
- [ ] Distributed training with 2 workers produces same result as single-worker (tolerance 1e-5)
- [ ] Dask example runs on localhost cluster without error
- [ ] All benchmark outputs recorded and committed to `benchmark/results/`
- [ ] Plugin system: `example_anisotropic_expand` compiles and loads at runtime
- [ ] No regression on v1 + v2 + v3 exit criteria
- [ ] `doc/architecture.md` updated to reflect learned DTDO and GPU paths
