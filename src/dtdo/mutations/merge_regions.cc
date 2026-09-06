#include "src/field/topology/topology_ops.h"
#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"


namespace vbx {

FieldState ApplyMergeRegions(const FieldState& s, vbx_region_id r1, vbx_region_id r2) {
    return MergeRegions(s, r1, r2);
}

} // namespace vbx
