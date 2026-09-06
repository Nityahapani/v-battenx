#pragma once
#include "vbatten_x/base.h"
#include "src/field/field_state.h"

namespace vbx {

FieldState Expand1dTo2d(const FieldState& s, vbx_region_id rid, uint64_t seed = 42);
FieldState Expand2dTo3d(const FieldState& s, vbx_region_id rid, uint64_t seed = 42);
FieldState Collapse3dTo2d(const FieldState& s, vbx_region_id rid);
FieldState Collapse2dTo1d(const FieldState& s, vbx_region_id rid);
FieldState ApplySplitRegion(const FieldState& s, vbx_region_id rid, int axis = 0);
FieldState ApplyMergeRegions(const FieldState& s, vbx_region_id r1, vbx_region_id r2);
FieldState ApplyAddConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2);
FieldState ApplyRemoveConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2);
FieldState ApplyLocalDimChange(const FieldState& s, vbx_region_id rid,
                                vbx_dim_t new_dim, uint64_t seed = 42);

} // namespace vbx
