#include "src/field/field_state.h"
#include "src/field/topology/topology_ops.cc"

namespace vbx {
FieldState ApplyAddConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2) {
    return AddConnection(s, r1, r2);
}
} // namespace vbx
