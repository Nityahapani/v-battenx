#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/dtdo/rule_based/complexity_pruner.cc"
#include "src/dtdo/rule_based/gradient_operator.cc"
#include "src/dtdo/rule_based/threshold_operator.cc"
#include <memory>
#include <vector>

namespace vbx {

class MutationRouter : public TopologicalOperator {
public:
    MutationRouter(float tau_expand    = 0.1f,
                   float tau_collapse  = 0.01f,
                   float split_thresh  = 0.05f,
                   float merge_thresh  = 0.005f,
                   int   prune_every   = 5)
    {
        pruner_    = MakeComplexityPruner(prune_every);
        gradient_  = MakeGradientOperator(split_thresh, merge_thresh);
        threshold_ = MakeThresholdOperator(tau_expand, tau_collapse);
    }

    MutationResult Apply(const FieldState&     current,
                          const ResidualInfo&  residuals,
                          const ComplexityCost& budget) const override {
        if (budget.OverBudget()) {
            auto r = pruner_->Apply(current, residuals, budget);
            if (r.mutated) return r;
        }

        auto r = gradient_->Apply(current, residuals, budget);
        if (r.mutated) return r;

        return threshold_->Apply(current, residuals, budget);
    }

    std::string Name() const override { return "router"; }

private:
    std::unique_ptr<TopologicalOperator> pruner_;
    std::unique_ptr<TopologicalOperator> gradient_;
    std::unique_ptr<TopologicalOperator> threshold_;
};

std::unique_ptr<TopologicalOperator> MakeMutationRouter(
    float tau_expand, float tau_collapse,
    float split_thresh, float merge_thresh,
    int prune_every)
{
    return std::make_unique<MutationRouter>(
        tau_expand, tau_collapse, split_thresh, merge_thresh, prune_every);
}

} // namespace vbx
