#pragma once

#include "base.h"
#include <string>
#include <vector>

namespace vbx {

enum class MutationType : int {
    NoOp           = 0,
    Expand1dTo2d   = 1,
    Expand2dTo3d   = 2,
    Collapse3dTo2d = 3,
    Collapse2dTo1d = 4,
    SplitRegion    = 5,
    MergeRegions   = 6,
    AddConnection  = 7,
    RemoveConnection = 8,
    LocalDimChange = 9,
};

inline const char* MutationTypeName(MutationType t) {
    switch (t) {
        case MutationType::NoOp:            return "no_op";
        case MutationType::Expand1dTo2d:    return "expand_1d_to_2d";
        case MutationType::Expand2dTo3d:    return "expand_2d_to_3d";
        case MutationType::Collapse3dTo2d:  return "collapse_3d_to_2d";
        case MutationType::Collapse2dTo1d:  return "collapse_2d_to_1d";
        case MutationType::SplitRegion:     return "split_region";
        case MutationType::MergeRegions:    return "merge_regions";
        case MutationType::AddConnection:   return "add_connection";
        case MutationType::RemoveConnection:return "remove_connection";
        case MutationType::LocalDimChange:  return "local_dim_change";
    }
    return "unknown";
}

struct MutationEvent {
    MutationType  type;
    vbx_region_id region_a;
    vbx_region_id region_b;
    vbx_dim_t     from_dim;
    vbx_dim_t     to_dim;
    float         residual_before;
    int           stage;
};

struct MutationLog {
    std::vector<MutationEvent> events;

    void Record(MutationType t, vbx_region_id ra, vbx_region_id rb,
                vbx_dim_t fd, vbx_dim_t td, float res, int stage) {
        events.push_back({t, ra, rb, fd, td, res, stage});
    }

    bool Empty() const { return events.empty(); }
    void Clear()       { events.clear(); }
};

} // namespace vbx
