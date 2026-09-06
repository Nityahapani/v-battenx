#include "vbatten_x/dimension.h"
#include <cmath>

namespace vbx {

float TotalDimCost(const DimensionMap& d, std::size_t num_regions) {
    float cost = 0.0f;
    for (std::size_t r = 0; r < num_regions; ++r) {
        float dim = static_cast<float>(d.LocalDim(static_cast<vbx_region_id>(r)));
        cost += dim * dim;
    }
    return cost;
}

float DimBudgetUsagePct(const DimensionMap& d, std::size_t num_regions) {
    float used   = d.BudgetUsed();
    float budget = d.DimBudget();
    return budget > 0.0f ? (used / budget) * 100.0f : 0.0f;
}

} // namespace vbx
