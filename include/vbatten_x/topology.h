#pragma once

#include "base.h"
#include <vector>
#include <memory>

namespace vbx {

struct RegionEdge {
    vbx_region_id src;
    vbx_region_id dst;
    vbx_float     weight;
};

class FieldTopology {
public:
    virtual ~FieldTopology() = default;

    virtual std::size_t               NumRegions()   const = 0;
    virtual std::vector<vbx_region_id> Neighbours(vbx_region_id r) const = 0;
    virtual std::vector<RegionEdge>   Edges()        const = 0;
    virtual bool                      IsConnected()  const = 0;

    virtual vbx_region_id AddRegion() = 0;
    virtual void          RemoveRegion(vbx_region_id r) = 0;
    virtual void          AddEdge(vbx_region_id a, vbx_region_id b, vbx_float w = 1.0f) = 0;
    virtual void          RemoveEdge(vbx_region_id a, vbx_region_id b) = 0;

    virtual std::unique_ptr<FieldTopology> Clone() const = 0;
};

} // namespace vbx
