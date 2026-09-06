#include "vbatten_x/dimension.h"
#include <unordered_map>
#include <stdexcept>
#include <numeric>
#include <cmath>

namespace vbx {

class AdaptiveDim : public DimensionMap {
public:
    AdaptiveDim(vbx_dim_t default_dim, float budget)
        : default_dim_(default_dim), budget_(budget) {}

    vbx_dim_t LocalDim(vbx_region_id r) const override {
        auto it = dims_.find(r);
        return it != dims_.end() ? it->second : default_dim_;
    }

    void SetDim(vbx_region_id r, vbx_dim_t d) override {
        if (d < 1) throw std::runtime_error("Dimension must be >= 1");
        dims_[r] = d;
    }

    float DimBudget() const override { return budget_; }

    float BudgetUsed() const override {
        float total = 0.0f;
        for (auto& [r, d] : dims_) total += static_cast<float>(d);
        return total;
    }

    std::unordered_map<vbx_region_id, vbx_dim_t> AllDims() const { return dims_; }

    float AvgDim() const {
        if (dims_.empty()) return static_cast<float>(default_dim_);
        float s = 0.0f;
        for (auto& [_, d] : dims_) s += d;
        return s / dims_.size();
    }

    float DimVariance() const {
        if (dims_.size() < 2) return 0.0f;
        float mean = AvgDim();
        float var  = 0.0f;
        for (auto& [_, d] : dims_) {
            float diff = d - mean;
            var += diff * diff;
        }
        return var / dims_.size();
    }

    std::unique_ptr<DimensionMap> Clone() const override {
        auto c = std::make_unique<AdaptiveDim>(default_dim_, budget_);
        c->dims_ = dims_;
        return c;
    }

private:
    vbx_dim_t                                     default_dim_;
    float                                          budget_;
    std::unordered_map<vbx_region_id, vbx_dim_t>  dims_;
};

std::unique_ptr<DimensionMap> MakeAdaptiveDim(vbx_dim_t default_dim, float budget) {
    return std::make_unique<AdaptiveDim>(default_dim, budget);
}

} // namespace vbx
