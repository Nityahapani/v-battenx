#include "src/field/topology/topology_ops.h"
#include "src/field/field_state.h"


namespace vbx {
FieldState ApplyRemoveConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2) {
    return RemoveConnection(s, r1, r2);
}
} // namespace vbx
