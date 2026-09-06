# V-BATTEN-X · v3 — The DTDO: Dynamic Topological-Dimensional Operator

> **Status: ✅ COMPLETE**
> 52/52 tests passing (41 v1+v2 regression + 11 new v3 tests).
> 6 C++ unit tests for dim transitions (all passing, zero energy loss).

> **Goal:** Implement the core algorithmic contribution. The DTDO restructures the
> field between boosting stages — changing dimensions, splitting/merging regions,
> rewiring topology — in response to physics residuals.

**Target:** Research preview. Topology evolution logged, queryable, and demoed.
**Depends on:** v2 ✅

---

## 1 · Core DTDO Abstraction

- [x] `include/vbatten_x/mutation_result.h`:
  - [x] `MutationType` enum — all 10: `NoOp`, `Expand1dTo2d`, `Expand2dTo3d`, `Collapse3dTo2d`, `Collapse2dTo1d`, `SplitRegion`, `MergeRegions`, `AddConnection`, `RemoveConnection`, `LocalDimChange`
  - [x] `MutationEvent` struct: type, region_a, region_b, from_dim, to_dim, residual_before, stage
  - [x] `MutationLog`: vector of events + `Record()`, `Clear()`, `Empty()`
  - [x] `MutationTypeName()` — string name for each type
- [x] `include/vbatten_x/topological_operator.h`:
  - [x] `ComplexityCost` — max_total_dim, max_regions, max_connections; `FromState()`, `CanAfford()`, `OverBudget()`, `BudgetRemaining()`
  - [x] `MutationResult` — `FieldState next` + `MutationLog` + `bool mutated`
  - [x] `TopologicalOperator` ABC — `Apply(current, residuals, budget) → MutationResult`, `Name()`

## 2 · Dimension Layer — Full Implementation

- [x] `src/field/dimension/adaptive_dim.cc` — `AdaptiveDim`: per-region `unordered_map<region_id, dim>`, `AllDims()`, `AvgDim()`, `DimVariance()`
- [x] `src/field/dimension/dim_transitions.h` — forward declarations (no ODR issues)
- [x] `src/field/dimension/dim_transitions.cc` — PCA-based transitions:
  - [x] `Expand1dTo2d` — appends near-zero noise axis, Frobenius norm preserved
  - [x] `Expand2dTo3d` — same principle for third axis
  - [x] `Collapse3dTo2d` — PCA top-2 components
  - [x] `Collapse2dTo1d` — PCA top-1 component
  - [x] `LocalDimChange(params, from_d, to_d, seed)` — general n→m, delegates to above for common cases, zero-padding with PCA for arbitrary transitions
  - [x] All transitions preserve Frobenius norm (verified: rel_err = 0.000000)
- [x] `src/field/dimension/dim_cost.cc` — `TotalDimCost()` (quadratic), `DimBudgetUsagePct()`

## 3 · Topology Layer — Full Implementation

- [x] `src/field/topology/simplex_complex.h` — header-only `SimplexComplex`: `Betti0()`, `Betti1()` (Euler characteristic), `GraphDiameter()`, `EdgeDensity()`
- [x] `src/field/topology/topology_ops.h` — forward declarations
- [x] `src/field/topology/topology_ops.cc` — `SplitRegion`, `MergeRegions`, `AddConnection`, `RemoveConnection` on `FieldState`
- [x] `src/field/topology/topology_cost.cc` — `BettiCost`, `DiameterCost`, `DensityCost`, `TotalTopologyCost`

## 4 · Atomic Mutations (`src/dtdo/mutations/`)

- [x] `mutations.h` — internal forward declarations
- [x] `expand_dim.cc` — `Expand1dTo2d`, `Expand2dTo3d` on `FieldState`
- [x] `collapse_dim.cc` — `Collapse3dTo2d`, `Collapse2dTo1d` on `FieldState`
- [x] `split_region.cc` — `ApplySplitRegion`
- [x] `merge_regions.cc` — `ApplyMergeRegions`
- [x] `add_edge.cc` — `ApplyAddConnection`
- [x] `remove_edge.cc` — `ApplyRemoveConnection`
- [x] `local_dim_change.cc` — `ApplyLocalDimChange`

## 5 · Tensor Adaptation

- [x] `src/field/tensor/rank_adaptive_tensor.cc` — `RankAdaptiveTensor`: identity-initialised, `AdaptToDimension()` copies top-left block + expands with identity

## 6 · ComplexityCost

- [x] `src/dtdo/complexity_cost.cc` — `ComplexityCost::FromState()` reads live regions, edges, sum of local dims from `FieldState`

## 7 · Rule-Based DTDO (`src/dtdo/rule_based/`)

- [x] `threshold_operator.cc` — expand if pde_r > τ_expand, collapse if pde_r < τ_collapse
- [x] `gradient_operator.cc` — split on high residual gradient, merge on low
- [x] `complexity_pruner.cc` — collapse every N stages or when over budget
- [x] `mutation_router.cc` — priority chain: Pruner → Gradient → Threshold
- [x] `operator_registry.cc` — `MakeOperator(name, ...)` factory: `"threshold"`, `"gradient"`, `"pruner"`, `"router"`, `"none"`

## 8 · Metrics

- [x] `src/metric/topology_metric.cc` — `TopologySnapshot`, `ComputeTopologySnapshot`, `TopologyDelta`
- [x] `src/metric/dimension_metric.cc` — `DimMetrics`: avg_local_dim, dim_variance, budget_usage_pct

## 9 · Learner v3

- [x] `src/learner.cc` — DTDO wired between `Eval` and `DoBoost`:
  1. `PhysicsEvaluator.Eval` → `ResidualInfo`
  2. `ComplexityCost::FromState` → budget
  3. `DTDO.Apply(state, residuals, budget)` → `MutationResult`
  4. If mutated: update `current_state`, record log
  5. `Objective.GetGradients` (+ PDE term if lambda_pde > 0)
  6. `LinearBooster.DoBoost`
  7. Accumulate predictions
- [x] `VariationalEnsemble` replaces `Ensemble` — stores `MutationLog` + PDE residuals per stage
- [x] `Save()` writes v3 format: `total_mutations`, per-stage `mutations`, `pde_before`, `pde_after`
- [x] `Load()` reads v3 format (backward compat with v2 fields)
- [x] Training log prints: loss, pde_r, avg_dim, dim_budget%, total_mutations (when verbose ≥ 2)
- [x] `dtdo`, `tau_expand`, `tau_collapse`, `max_total_dim`, `max_regions`, `max_connections` params

## 10 · Python API

- [x] `core.py` — `MutationEvent` class; `Booster.get_mutation_log()`, `Booster.get_stage_pde_residuals()`
- [x] `sklearn.py` — both estimators gain `dtdo`, `tau_expand`, `tau_collapse`, `max_total_dim`, `max_regions` params; `mutation_log_` property
- [x] `__init__.py` — exports `MutationEvent`; version `3.0.0`
- [x] `version.h`, `pyproject.toml`, `setup.cfg` — bumped to `3.0.0`

## 11 · Build

- [x] `src/field/topology/simplex_complex.h` — promoted to header-only (was causing ODR violations when included via .cc chain)
- [x] `src/field/topology/topology_ops.h` / `src/field/dimension/dim_transitions.h` — forward-declaration headers added
- [x] `src/vbatten_x_impl.cc` — unity build updated with all v3 sources; each `.cc` included exactly once

## 12 · Tests

- [x] `tests/cpp/test_dim_transitions.cc` — 6 C++ tests: Frobenius norm preservation for all expand/collapse variants; all pass with rel_err = 0.000000
- [x] `tests/python/test_training.py` — 7 tests: includes `dtdo=threshold`, `dtdo=router`, `dtdo=none` equivalence
- [x] `tests/python/test_dtdo.py` — 8 tests: MutationEvent, log access, PDE residuals, JSON version, sklearn interface, budget params, save/load

## 13 · Docs & Demo

- [x] `demo/guide-python/03_topology_visualization.py` — plain vs DTDO comparison, mutation log printout
- [x] `doc/dtdo.md` — mutation types table, dim transitions, all strategies, complexity budget, tuning guide, v3 model format

## 14 · v3 Exit Criteria — Results

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| pytest suite | all pass | 52/52 | ✅ |
| C++ dim transition tests | 6/6 pass, norm err < 5% | 6/6, err = 0% | ✅ |
| DTDO fires on real data | log contains events | confirmed | ✅ |
| `dtdo=none` ≡ plain | predictions identical | max diff = 0 | ✅ |
| Budget params respected | training completes | passes | ✅ |
| Save/load with DTDO | bit-exact roundtrip | max diff = 0 | ✅ |
| MutationLog in JSON | per-stage events | confirmed | ✅ |
| version string | 3.0.0 everywhere | ✅ | ✅ |
| v1+v2 regression | 41/41 | 41/41 | ✅ |
| doc/dtdo.md | written + reviewed | ✅ | ✅ |

## 15 · Known Carry-Forwards into v4

- Learned DTDO (neural network policy) — designed in v3 plan, implementation starts v4
- `test_topology_ops.cc` C++ unit tests (split→merge round-trip, Betti numbers on known graphs) — scheduled for v4 alongside GTest infrastructure
- DTDO currently operates on region 0 only (single-region field from v1) — full multi-region support lands in v4 when variational booster produces multi-region fields
- `AdaptiveDim` implemented but not yet used in learner loop — learner still uses `UniformDim`; switching to adaptive is a v4 task tied to multi-region support
