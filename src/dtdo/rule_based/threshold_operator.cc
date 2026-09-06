#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/dtdo/mutations/mutations.h"

namespace vbx {

class ThresholdOperator : public TopologicalOperator {
public:
    ThresholdOperator(float tau_expand = 0.1f, float tau_collapse = 0.01f)
        : tau_expand_(tau_expand), tau_collapse_(tau_collapse) {}

    MutationResult Apply(const FieldState&     current,
                          const ResidualInfo&  residuals,
                          const ComplexityCost& budget) const override {
        MutationResult result;
        result.next    = current.Clone();
        result.mutated = false;

        float pde_r = residuals.MeanPde();

        if (pde_r > tau_expand_ && budget.CanAfford(MutationType::Expand1dTo2d)) {
            vbx_dim_t dim = current.d->LocalDim(0);
            if (dim == 1) {
                result.next    = Expand1dTo2d(current, 0, seed_++);
                result.mutated = true;
                result.log.Record(MutationType::Expand1dTo2d, 0, 0, 1, 2, pde_r, 0);
            } else if (dim == 2 && budget.CanAfford(MutationType::Expand2dTo3d)) {
                result.next    = Expand2dTo3d(current, 0, seed_++);
                result.mutated = true;
                result.log.Record(MutationType::Expand2dTo3d, 0, 0, 2, 3, pde_r, 0);
            }
        } else if (pde_r < tau_collapse_) {
            vbx_dim_t dim = current.d->LocalDim(0);
            if (dim == 3) {
                result.next    = Collapse3dTo2d(current, 0);
                result.mutated = true;
                result.log.Record(MutationType::Collapse3dTo2d, 0, 0, 3, 2, pde_r, 0);
            } else if (dim == 2) {
                result.next    = Collapse2dTo1d(current, 0);
                result.mutated = true;
                result.log.Record(MutationType::Collapse2dTo1d, 0, 0, 2, 1, pde_r, 0);
            }
        }

        return result;
    }

    std::string Name() const override { return "threshold"; }

private:
    float            tau_expand_;
    float            tau_collapse_;
    mutable uint64_t seed_ = 42;
};

std::unique_ptr<TopologicalOperator> MakeThresholdOperator(float tau_expand,
                                                             float tau_collapse) {
    return std::make_unique<ThresholdOperator>(tau_expand, tau_collapse);
}

} // namespace vbx
