#include "src/field/field_impls.h"

namespace vbx {

class UniformDim : public DimensionMap {
public:
    UniformDim(vbx_dim_t dim, std::size_t num_regions, float budget)
        : dim_(dim), num_regions_(num_regions), budget_(budget) {}

    vbx_dim_t LocalDim(vbx_region_id) const override { return dim_; }
    void      SetDim(vbx_region_id, vbx_dim_t d)    override { dim_ = d; }
    float     DimBudget()  const override { return budget_; }
    float     BudgetUsed() const override { return static_cast<float>(dim_ * num_regions_); }

    std::unique_ptr<DimensionMap> Clone() const override {
        return std::make_unique<UniformDim>(dim_, num_regions_, budget_);
    }

private:
    vbx_dim_t   dim_;
    std::size_t num_regions_;
    float       budget_;
};

std::unique_ptr<DimensionMap> MakeUniformDim(vbx_dim_t dim, std::size_t num_regions, float budget) {
    return std::make_unique<UniformDim>(dim, num_regions, budget);
}

} // namespace vbx
