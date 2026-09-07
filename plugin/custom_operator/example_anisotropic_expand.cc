#include "plugin/custom_operator/custom_operator_interface.h"
#include "src/dtdo/mutations/mutations.h"
#include <Eigen/Dense>
#include <cmath>

namespace vbx {

class AnisotropicExpandOperator : public TopologicalOperator {
public:
    AnisotropicExpandOperator(float residual_threshold = 0.05f)
        : threshold_(residual_threshold), seed_(42) {}

    MutationResult Apply(const FieldState&     current,
                          const ResidualInfo&  residuals,
                          const ComplexityCost& budget) const override {
        MutationResult result;
        float pde_r = residuals.MeanPde();

        if (pde_r <= threshold_ || !budget.CanAfford(MutationType::Expand1dTo2d)) {
            result.next    = current.Clone();
            result.mutated = false;
            return result;
        }

        auto params = current.F->Params();
        if (params.empty()) {
            result.next    = current.Clone();
            result.mutated = false;
            return result;
        }

        // Expand only along the direction of maximum residual gradient
        // approximated by the parameter with the largest absolute value
        std::size_t max_idx = 0;
        float       max_val = 0.0f;
        for (std::size_t i = 0; i < params.size(); ++i) {
            float v = std::abs(params[i]);
            if (v > max_val) { max_val = v; max_idx = i; }
        }

        std::vector<float> expanded(params.size() * 2, 0.0f);
        for (std::size_t i = 0; i < params.size(); ++i) {
            expanded[2 * i]     = params[i];
            expanded[2 * i + 1] = (i == max_idx) ? pde_r * 0.1f : 0.0f;
        }

        result.next = current.Clone();
        result.next.F->Embed({expanded.data(), expanded.size()});
        result.next.d->SetDim(0, 2);
        result.next.T->AdaptToDimension(2);
        result.mutated = true;
        result.log.Record(MutationType::Expand1dTo2d, 0, 0,
                           current.d->LocalDim(0), 2, pde_r, 0);
        return result;
    }

    std::string Name() const override { return "anisotropic_expand"; }

private:
    float            threshold_;
    mutable uint64_t seed_;
};

static std::unique_ptr<TopologicalOperator> MakeAnisotropicExpand() {
    return std::make_unique<AnisotropicExpandOperator>();
}

} // namespace vbx

VBATTENX_REGISTER_OPERATOR(
    "anisotropic_expand",
    vbx::MakeAnisotropicExpand,
    "Expands dimension only along direction of max PDE residual gradient")
