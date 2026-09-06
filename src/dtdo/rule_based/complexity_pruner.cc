#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"
#include "src/dtdo/mutations/mutations.h"

namespace vbx {

class ComplexityPruner : public TopologicalOperator {
public:
    explicit ComplexityPruner(int prune_every = 5) : prune_every_(prune_every) {}

    MutationResult Apply(const FieldState&     current,
                          const ResidualInfo&  residuals,
                          const ComplexityCost& budget) const override {
        MutationResult result;
        result.mutated = false;
        ++call_count_;

        if (!budget.OverBudget() && call_count_ % prune_every_ != 0) {
            result.next = current.Clone();
            return result;
        }

        vbx_dim_t dim = current.d->LocalDim(0);
        if (dim > 1) {
            if (dim == 3) {
                result.next    = Collapse3dTo2d(current, 0);
                result.log.Record(MutationType::Collapse3dTo2d, 0, 0, 3, 2,
                                   residuals.MeanPde(), 0);
            } else {
                result.next    = Collapse2dTo1d(current, 0);
                result.log.Record(MutationType::Collapse2dTo1d, 0, 0, 2, 1,
                                   residuals.MeanPde(), 0);
            }
            result.mutated = true;
        } else {
            result.next = current.Clone();
        }

        return result;
    }

    std::string Name() const override { return "complexity_pruner"; }

private:
    int              prune_every_;
    mutable int      call_count_ = 0;
};

std::unique_ptr<TopologicalOperator> MakeComplexityPruner(int prune_every) {
    return std::make_unique<ComplexityPruner>(prune_every);
}

} // namespace vbx
