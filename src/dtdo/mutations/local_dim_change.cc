#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include "src/field/dimension/dim_transitions.cc"

namespace vbx {

FieldState ApplyLocalDimChange(const FieldState& s, vbx_region_id rid,
                                vbx_dim_t new_dim, uint64_t seed) {
    vbx_dim_t old_dim = s.d->LocalDim(rid);
    if (old_dim == new_dim) return s.Clone();

    FieldState out  = s.Clone();
    auto params     = s.F->Params();
    auto new_params = LocalDimChange(params, old_dim, new_dim, seed);

    out.F = MakeContinuousField(new_params.size(), 1);
    out.F->Embed({new_params.data(), new_params.size()});
    out.d->SetDim(rid, new_dim);
    out.T->AdaptToDimension(new_dim);
    return out;
}

} // namespace vbx
