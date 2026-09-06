#include "src/field/dimension/dim_transitions.h"
#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"

#include <random>

namespace vbx {

FieldState Expand1dTo2d(const FieldState& s, vbx_region_id rid, uint64_t seed) {
    FieldState out = s.Clone();
    auto params    = s.F->Params();
    auto expanded  = Expand1dTo2d(params, seed);
    out.F->Embed({expanded.data(), expanded.size()});
    out.d->SetDim(rid, 2);
    out.T->AdaptToDimension(2);
    return out;
}

FieldState Expand2dTo3d(const FieldState& s, vbx_region_id rid, uint64_t seed) {
    FieldState out = s.Clone();
    auto params    = s.F->Params();
    auto expanded  = Expand2dTo3d(params, seed);
    out.F->Embed({expanded.data(), expanded.size()});
    out.d->SetDim(rid, 3);
    out.T->AdaptToDimension(3);
    return out;
}

} // namespace vbx
