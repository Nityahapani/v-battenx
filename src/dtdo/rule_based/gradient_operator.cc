#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/dtdo/mutations/mutations.h"
#include <cmath>

namespace vbx {

class GradientOperator : public TopologicalOperator {
public:
    GradientOperator(float split_threshold = 0.05f, float merge_threshold = 0.005f)
        : split_thresh_(split_threshold), merge_thresh_(merge_threshold) {}

    MutationResult Apply(const FieldState&     current,
                          const ResidualInfo&  residuals,
                          const ComplexityCost& budget) const override {
        MutationResult result;
        result.mutated = false;

        float pde_r = residuals.MeanPde();
        float grad_mag = pde_r;

        if (grad_mag > split_thresh_
            && budget.CanAfford(MutationType::SplitRegion)
            && current.K->NumRegions() < static_cast<std::size_t>(budget.max_regions)) {
            result.next    = ApplySplitRegion(current, 0, 0);
            result.mutated = true;
            result.log.Record(MutationType::SplitRegion, 0, 1, 0, 0, pde_r, 0);
        } else if (grad_mag < merge_thresh_
                   && current.K->NumRegions() > 1) {
            auto edges = current.K->Edges();
            if (!edges.empty()) {
                result.next    = ApplyMergeRegions(current, edges[0].src, edges[0].dst);
                result.mutated = true;
                result.log.Record(MutationType::MergeRegions,
                                   edges[0].src, edges[0].dst, 0, 0, pde_r, 0);
            } else {
                result.next = current.Clone();
            }
        } else {
            result.next = current.Clone();
        }

        return result;
    }

    std::string Name() const override { return "gradient"; }

private:
    float split_thresh_;
    float merge_thresh_;
};

std::unique_ptr<TopologicalOperator> MakeGradientOperator(float split_thresh,
                                                            float merge_thresh) {
    return std::make_unique<GradientOperator>(split_thresh, merge_thresh);
}

} // namespace vbx
