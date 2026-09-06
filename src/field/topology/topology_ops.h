#pragma once
#include "vbatten_x/base.h"
#include "src/field/field_state.h"

namespace vbx {
FieldState SplitRegion(const FieldState& s, vbx_region_id rid, int axis = 0);
FieldState MergeRegions(const FieldState& s, vbx_region_id r1, vbx_region_id r2);
FieldState AddConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2);
FieldState RemoveConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2);
}
