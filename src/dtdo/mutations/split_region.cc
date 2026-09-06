#include "src/field/topology/topology_ops.h"
#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"


namespace vbx {

FieldState ApplySplitRegion(const FieldState& s, vbx_region_id rid, int axis) {
    return SplitRegion(s, rid, axis);
}

} // namespace vbx
