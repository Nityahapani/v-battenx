#pragma once

#include "base.h"
#include <memory>

namespace vbx {

class DimensionMap {
public:
    virtual ~DimensionMap() = default;

    virtual vbx_dim_t LocalDim(vbx_region_id r) const = 0;
    virtual void      SetDim(vbx_region_id r, vbx_dim_t d) = 0;
    virtual float     DimBudget() const = 0;
    virtual float     BudgetUsed() const = 0;

    virtual std::unique_ptr<DimensionMap> Clone() const = 0;
};

} // namespace vbx
