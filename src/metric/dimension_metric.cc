#include "vbatten_x/dimension.h"
#include "vbatten_x/base.h"
#include <cmath>

namespace vbx {

struct DimMetrics {
    float avg_local_dim;
    float dim_variance;
    float budget_usage_pct;
};

DimMetrics ComputeDimMetrics(const DimensionMap& d, std::size_t num_regions) {
    if (num_regions == 0) return {0.0f, 0.0f, 0.0f};

    float sum = 0.0f;
    for (std::size_t r = 0; r < num_regions; ++r)
        sum += static_cast<float>(d.LocalDim(static_cast<vbx_region_id>(r)));
    float avg = sum / static_cast<float>(num_regions);

    float var = 0.0f;
    for (std::size_t r = 0; r < num_regions; ++r) {
        float diff = static_cast<float>(d.LocalDim(static_cast<vbx_region_id>(r))) - avg;
        var += diff * diff;
    }
    var /= static_cast<float>(num_regions);

    float budget_pct = d.DimBudget() > 0.0f
                       ? (d.BudgetUsed() / d.DimBudget()) * 100.0f
                       : 0.0f;

    return {avg, var, budget_pct};
}

} // namespace vbx
