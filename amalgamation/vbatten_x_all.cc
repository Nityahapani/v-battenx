// V-BATTEN-X — single-file amalgamation build
// Generated for v5.0.0
//
// Usage:
//   g++ -std=c++17 -O2 -shared -fPIC \
//       -I/path/to/vbatten_x/include \
//       -I/usr/include/eigen3 \
//       vbatten_x_all.cc \
//       -o libvbatten_x.so
//
// All source files are included once in dependency order.
// No ODR violations — unity build pattern.

// ── common ────────────────────────────────────────────────────────────────────
#include "../src/common/json.cc"

// ── field layer ──────────────────────────────────────────────────────────────
#include "../src/field/field_state.cc"
#include "../src/field/manifold/continuous_field.cc"
#include "../src/field/topology/region_graph.cc"
#include "../src/field/topology/topology_ops.cc"
#include "../src/field/topology/topology_cost.cc"
#include "../src/field/dimension/uniform_dim.cc"
#include "../src/field/dimension/adaptive_dim.cc"
#include "../src/field/dimension/dim_transitions.cc"
#include "../src/field/dimension/dim_cost.cc"
#include "../src/field/tensor/contraction_engine.cc"
#include "../src/field/tensor/rank_adaptive_tensor.cc"

// ── data ──────────────────────────────────────────────────────────────────────
#include "../src/data/physical_dataset.cc"
#include "../src/data/normalizer.cc"
#include "../src/data/physics_meta.cc"
#include "../src/data/partitioner.cc"

// ── physics ───────────────────────────────────────────────────────────────────
#include "../src/physics/residual_info.cc"
#include "../src/physics/pde/finite_diff_ops.cc"
#include "../src/physics/pde/conservation_ops.cc"
#include "../src/physics/evaluator_registry.cc"

// ── objective & metric ────────────────────────────────────────────────────────
#include "../src/objective/regression_obj.cc"
#include "../src/objective/classification_obj.cc"
#include "../src/objective/pde_constrained_obj.cc"
#include "../src/objective/physics_informed_obj.cc"
#include "../src/metric/prediction_metric.cc"
#include "../src/metric/pde_metric.cc"

// ── predictor ─────────────────────────────────────────────────────────────────
#include "../src/predictor/cpu_predictor.cc"

// ── encoder ───────────────────────────────────────────────────────────────────
#include "../src/encoder/mlp_encoder.cc"
#include "../src/encoder/fourier_encoder.cc"

// ── DTDO ──────────────────────────────────────────────────────────────────────
#include "../src/dtdo/complexity_cost.cc"
#include "../src/dtdo/mutations/expand_dim.cc"
#include "../src/dtdo/mutations/collapse_dim.cc"
#include "../src/dtdo/mutations/split_region.cc"
#include "../src/dtdo/mutations/merge_regions.cc"
#include "../src/dtdo/mutations/add_edge.cc"
#include "../src/dtdo/mutations/remove_edge.cc"
#include "../src/dtdo/mutations/local_dim_change.cc"
#include "../src/dtdo/operator_registry.cc"

// ── booster ───────────────────────────────────────────────────────────────────
#include "../src/booster/linear/linear_booster.cc"

// ── distributed ───────────────────────────────────────────────────────────────
#include "../src/collective/local_communicator.cc"

// ── learner + C ABI ──────────────────────────────────────────────────────────
#include "../src/learner.cc"
#include "../src/c_api.cc"
