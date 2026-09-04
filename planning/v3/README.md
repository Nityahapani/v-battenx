# V-BATTEN-X · v3 — The DTDO: Dynamic Topological-Dimensional Operator

> **Goal:** Implement the core algorithmic contribution of V-BATTEN-X.
> The DTDO is the component that lets the model _restructure itself_ — changing
> dimensions, splitting/merging regions, rewiring topology — in response to
> physics residuals. This is what separates V-BATTEN-X from all existing boosting libraries.
>
> v3 ships the **rule-based DTDO** (heuristic, deterministic, fast) as the primary
> strategy. The learned DTDO (neural network policy) is designed here and begins
> in v4. Both share the same interface and are interchangeable.

**Target:** Research preview. Topology evolution is visible, logged, and visualisable.
**Depends on:** v2 complete. PDE residuals must be real numbers for DTDO to act on.

---

## 1 · The Core DTDO Abstraction

### 1a · Public Interface (`include/vbatten_x/topological_operator.h`)

- [ ] `TopologicalOperator` ABC — single method:
  ```cpp
  MutationResult Apply(
    const FieldState&    current,
    const ResidualInfo&  residuals,
    const ComplexityCost& budget
  ) const;
  ```
- [ ] `MutationResult` (`mutation_result.h`):
  - [ ] `FieldState next` — the mutated state
  - [ ] `MutationLog log` — what changed, where, why
  - [ ] `bool mutated` — false if NO_OP
- [ ] `MutationType` enum — **all 10 mutation types**:
  - [ ] `EXPAND_1D_TO_2D` — embed a 1D region into 2D space
  - [ ] `EXPAND_2D_TO_3D` — embed a 2D region into 3D space
  - [ ] `COLLAPSE_3D_TO_2D` — project 3D region down to 2D
  - [ ] `COLLAPSE_2D_TO_1D` — project 2D region down to 1D
  - [ ] `SPLIT_REGION` — divide one region into two at a learned boundary
  - [ ] `MERGE_REGIONS` — fuse two adjacent low-residual regions
  - [ ] `ADD_CONNECTION` — add edge between non-adjacent regions
  - [ ] `REMOVE_CONNECTION` — remove edge between regions
  - [ ] `LOCAL_DIM_CHANGE` — change dimension of exactly one region (n → m)
  - [ ] `NO_OP` — leave field unchanged

### 1b · Complexity Budget (`ComplexityCost`)

- [ ] `ComplexityCost` struct:
  - [ ] `max_total_dim` — hard cap on sum of all `local_dim(r)` values
  - [ ] `max_regions` — hard cap on number of regions
  - [ ] `max_connections` — hard cap on topology edges
  - [ ] `current_usage()` — compute from current `FieldState`
  - [ ] `budget_remaining()` — headroom before hitting any cap
  - [ ] `can_afford(MutationType)` — bool: would this mutation exceed budget?

## 2 · Topology Layer — Full Implementation (`src/field/topology/`)

> v1 had a single-region stub. v3 needs the full multi-region graph.

- [ ] `region_graph.cc` — full adjacency graph:
  - [ ] `add_region(id, local_dim)` → `region_id`
  - [ ] `remove_region(id)` — also removes all incident edges
  - [ ] `add_edge(r1, r2, weight)` — directed or undirected
  - [ ] `remove_edge(r1, r2)`
  - [ ] `neighbours(r)` — list of adjacent region ids
  - [ ] `is_connected()` — graph connectivity check (BFS)
  - [ ] Backed by adjacency list (`std::unordered_map<region_id, std::vector<region_id>>`)
- [ ] `simplex_complex.cc` — simplicial complex on top of region graph:
  - [ ] Track 0-simplices (points), 1-simplices (edges), 2-simplices (triangles)
  - [ ] `betti_0()` — number of connected components
  - [ ] `betti_1()` — number of independent cycles
  - [ ] Used by `topology_metric.cc` to track topological complexity
- [ ] `topology_ops.cc`:
  - [ ] `split_region(state, region_id, split_axis)` → two child regions
    - [ ] Parent tensor projected onto two sub-tensors
    - [ ] New edge between children with weight 1.0
  - [ ] `merge_regions(state, r1, r2)` → one merged region
    - [ ] Tensors averaged (weighted by region size)
    - [ ] Edges of both regions unioned
  - [ ] `add_connection(state, r1, r2)` → updated topology
  - [ ] `remove_connection(state, r1, r2)` → updated topology
- [ ] `topology_cost.cc`:
  - [ ] `betti_cost()` — penalty proportional to `betti_1` (cycles are expensive)
  - [ ] `diameter_cost()` — graph diameter (long chains are expensive)
  - [ ] `density_cost()` — edge density relative to complete graph
- [ ] **Tests** (`test_topology_ops.cc`):
  - [ ] split → merge → original topology recovered
  - [ ] add_edge → remove_edge → original topology recovered
  - [ ] Betti numbers correct on known graphs (path, cycle, complete)

## 3 · Dimension Layer — Full Implementation (`src/field/dimension/`)

- [ ] `adaptive_dim.cc`:
  - [ ] `std::unordered_map<region_id, int> dim_map`
  - [ ] `local_dim(r)` → int
  - [ ] `set_dim(r, d)` — only valid if `d >= 1` and within budget
  - [ ] `all_dims()` → map of all regions
- [ ] `dim_transitions.cc` — **the actual math of changing dimension**:
  - [ ] `expand_1d_to_2d(tensor, basis)`:
    - [ ] Embed 1D vector into 2D by appending a new learned basis direction
    - [ ] New basis direction initialised via PCA of residuals in that region
  - [ ] `expand_2d_to_3d(tensor, basis)`:
    - [ ] Same principle — third axis = learned direction of max residual variance
  - [ ] `collapse_3d_to_2d(tensor)`:
    - [ ] Project via PCA: keep top-2 principal components
    - [ ] Track projection matrix for reverse operation
  - [ ] `collapse_2d_to_1d(tensor)`:
    - [ ] Project onto first principal component
  - [ ] `local_dim_change(tensor, from_d, to_d)`:
    - [ ] Generalises above: handles any n→m via PCA project/embed
  - [ ] All transitions preserve `‖tensor‖_F` approximately (normalise after)
- [ ] `dim_cost.cc`:
  - [ ] `total_dim_cost()` = Σ_r `local_dim(r)²` (quadratic penalty)
  - [ ] `dim_variance()` = variance of `local_dim` across regions
- [ ] **Tests** (`test_dim_transitions.cc`):
  - [ ] 1D → 2D → 1D: reconstruction error < 5% (PCA round-trip)
  - [ ] 2D → 3D → 2D: reconstruction error < 5%
  - [ ] `local_dim_change` agrees with composed expand/collapse for `n=1,2,3`

## 4 · Atomic Mutations (`src/dtdo/mutations/`)

> Each file implements one `MutationType`. All take a `FieldState` and return a new one.
> These are the building blocks the DTDO policy chooses from.

- [ ] `expand_dim.cc`:
  - [ ] `Expand1dTo2d(state, region_id) → FieldState`
  - [ ] `Expand2dTo3d(state, region_id) → FieldState`
  - [ ] Calls `dim_transitions::expand_*` on the tensor in that region
  - [ ] Updates `DimensionMap` in the returned state
- [ ] `collapse_dim.cc`:
  - [ ] `Collapse3dTo2d(state, region_id) → FieldState`
  - [ ] `Collapse2dTo1d(state, region_id) → FieldState`
- [ ] `split_region.cc`:
  - [ ] `SplitRegion(state, region_id, axis) → FieldState`
  - [ ] Calls `topology_ops::split_region`
  - [ ] Splits the tensor along `axis` (left half / right half)
- [ ] `merge_regions.cc`:
  - [ ] `MergeRegions(state, r1, r2) → FieldState`
  - [ ] Only valid if `r1` and `r2` are adjacent in topology
- [ ] `add_edge.cc` / `remove_edge.cc`:
  - [ ] Simple wrappers around `topology_ops::add_connection` / `remove_connection`
- [ ] `local_dim_change.cc`:
  - [ ] `LocalDimChange(state, region_id, new_dim) → FieldState`
  - [ ] Calls `dim_transitions::local_dim_change`
- [ ] **Tests** (`test_dtdo_mutations.cc`):
  - [ ] Each mutation in isolation: before/after invariants
  - [ ] `FieldState` total parameter count changes as expected
  - [ ] `MutationLog` records correct `MutationType` and `region_id`

## 5 · Rule-Based DTDO (`src/dtdo/rule_based/`)

> Deterministic heuristic policy. No training needed. Fast to run.
> This ships in v3 and serves as the production-ready baseline until the learned DTDO is ready.

- [ ] `threshold_operator.cc` — `ThresholdOperator`:
  - [ ] For each region `r`: if `pde_residual(r) > τ_expand` → expand dimension
  - [ ] If `pde_residual(r) < τ_collapse` AND `dim(r) > 1` → collapse dimension
  - [ ] `τ_expand` and `τ_collapse` are configurable parameters
  - [ ] After dimension change: record in `MutationLog`
- [ ] `gradient_operator.cc` — `GradientOperator`:
  - [ ] Compute residual gradient w.r.t. current field
  - [ ] High gradient magnitude → split region along gradient direction
  - [ ] Low gradient magnitude → candidate for merge
  - [ ] Prioritises splits in regions where `|∇residual|` is largest
- [ ] `complexity_pruner.cc` — `ComplexityPruner`:
  - [ ] If `budget_remaining() < 0` → prune:
    - [ ] Collapse the region with lowest `pde_residual` and dim > 1
    - [ ] Merge the two regions with lowest combined residual that are adjacent
  - [ ] Runs after every 5 boosting stages by default
- [ ] `mutation_router.cc` — **decides which operator fires for which region**:
  - [ ] Input: `ResidualInfo`, `ComplexityCost`, current `FieldState`
  - [ ] Priority order: ComplexityPruner (if over budget) → GradientOperator → ThresholdOperator
  - [ ] Returns a list of `(region_id, MutationType)` pairs
  - [ ] Applies them in order, passing updated state through
- [ ] `operator_registry.cc` — factory: `name → TopologicalOperator*`

## 6 · Tensor Adaptation After Mutation (`src/field/tensor/`)

> After the DTDO fires, tensors must adapt to the new dimensional structure.

- [ ] `rank_adaptive_tensor.cc`:
  - [ ] `adapt_to_dimension(region_id, new_dim)`:
    - [ ] If `new_dim > old_dim`: embed via `expand_*`
    - [ ] If `new_dim < old_dim`: project via `collapse_*`
    - [ ] If `new_dim == old_dim`: no-op
  - [ ] Tracks `rank` separately per region
- [ ] `contraction_engine.cc`:
  - [ ] Handle **ragged** tensor structures (different rank per region)
  - [ ] `contract_region(r, other)` — contraction respects `local_dim(r)`
  - [ ] Batch contraction across all regions

## 7 · Learner Integration (update `src/learner.cc`)

> Update the outer loop from v1 to call DTDO between boosting stages.

- [ ] Updated v3 sequence per iteration:
  - [ ] `PhysicsEncoder.Encode(dataset)` → `FieldState`
  - [ ] `PhysicsEvaluator.Eval(field_state)` → `ResidualInfo` (real PDE residuals now)
  - [ ] `TopologicalOperator.Apply(state, residuals, budget)` → `FieldState'` ← **new**
  - [ ] `TensorField.AdaptToDimension(new_d)` ← **new**
  - [ ] `Objective.GetGradients(pred, label)` → `(g, h)`
  - [ ] `VariationalBooster.DoBoost(dataset, g, h)` → new stage
  - [ ] `Ensemble.Append(stage)`
  - [ ] Convergence + budget check
- [ ] `MutationLog` stored in `Ensemble` alongside each `StageModel`
- [ ] Training log now includes: RMSE, PDE_residual, topology delta, dim usage

## 8 · Topology & Dimension Metrics (`src/metric/`)

- [ ] `topology_metric.cc`:
  - [ ] `TOPOLOGY_DELTA` — number of regions + edges added/removed this stage
  - [ ] `BETTI_NUMBER_CHANGE` — Δbetti_0, Δbetti_1 this stage
  - [ ] `CONNECTIVITY_COST` — graph diameter * density
- [ ] `dimension_metric.cc`:
  - [ ] `AVG_LOCAL_DIM` — mean dim across regions
  - [ ] `DIM_VARIANCE` — variance of local dims
  - [ ] `DIM_BUDGET_USAGE` — percentage of max_total_dim used

## 9 · Variational Booster — Stage Model (`src/booster/variational/`)

- [ ] `stage_model.cc`:
  - [ ] Owns one `(F, K, d, T)` snapshot at the time of boosting
  - [ ] Also stores `MutationLog` describing how this stage's topology was created
  - [ ] `Predict(dataset)` — field eval on this stage's structure
- [ ] `variational_booster.cc`:
  - [ ] Each stage: calls DTDO to mutate → fits residuals on new structure
  - [ ] Stage weight = `learning_rate * (1 - complexity_penalty)`
  - [ ] `ensemble.cc` — accumulates stage predictions with weights
- [ ] **This is the critical loop** — get it right before v4

## 10 · Visualisation — Topology Evolution

- [ ] `python-package/vbatten_x/field_viz.py`:
  - [ ] `plot_topology(stage)` — networkx graph of regions + connections, coloured by dim
  - [ ] `plot_topology_evolution(ensemble)` — animated GIF or step-by-step plot
  - [ ] `plot_dimension_map(stage)` — heatmap of `local_dim` per region
  - [ ] `plot_mutation_history(ensemble)` — timeline of what mutation fired when
- [ ] `demo/guide-python/03_topology_visualization.py` — run on heat equation data, show topology growing

## 11 · v3 Exit Criteria (Definition of Done)

- [ ] DTDO fires correctly on a 5-region field: topology changes are logged
- [ ] `test_dtdo_mutations.cc` — all 10 mutation types pass isolation tests
- [ ] `test_topology_ops.cc` — split→merge round-trip exact; Betti numbers correct
- [ ] `test_dim_transitions.cc` — 1D→2D→1D reconstruction error < 5%
- [ ] Learner outer loop calls DTDO every iteration without segfault (run 100 iters)
- [ ] `DIM_BUDGET_USAGE` metric reported in training log
- [ ] Topology evolution visualisation works: plot shows regions splitting over stages
- [ ] Rule-based DTDO reduces PDE residual by ≥ 10% vs no-DTDO baseline on heat data
- [ ] AddressSanitizer clean on full learner loop with DTDO enabled
- [ ] `doc/dtdo.md` — algorithmic deep-dive written and reviewed by team
- [ ] No regression on v1 + v2 exit criteria
