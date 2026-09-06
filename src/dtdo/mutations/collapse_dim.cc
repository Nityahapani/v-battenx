#include "src/field/dimension/dim_transitions.h"
#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"


namespace vbx {

FieldState Collapse3dTo2d(const FieldState& s, vbx_region_id rid) {
    FieldState out    = s.Clone();
    auto params       = s.F->Params();
    auto collapsed    = Collapse3dTo2d(params);
    out.F = MakeContinuousField(collapsed.size(), 1);
    out.F->Embed({collapsed.data(), collapsed.size()});
    out.d->SetDim(rid, 2);
    out.T->AdaptToDimension(2);
    return out;
}

FieldState Collapse2dTo1d(const FieldState& s, vbx_region_id rid) {
    FieldState out = s.Clone();
    auto params    = s.F->Params();
    auto collapsed = Collapse2dTo1d(params);
    out.F = MakeContinuousField(collapsed.size(), 1);
    out.F->Embed({collapsed.data(), collapsed.size()});
    out.d->SetDim(rid, 1);
    out.T->AdaptToDimension(1);
    return out;
}

} // namespace vbx
