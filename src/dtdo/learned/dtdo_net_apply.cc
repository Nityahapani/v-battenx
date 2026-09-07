#include "src/dtdo/learned/dtdo_net.cc"
#include "src/dtdo/mutations/mutations.h"

namespace vbx {

MutationResult DtdoNet::ApplyAction(const FieldState& s, const Action& act) const {
    MutationResult result;
    result.mutated = true;

    try {
        switch (act.type) {
            case MutationType::Expand1dTo2d:
                result.next = Expand1dTo2d(s, act.region, rng_());
                result.log.Record(act.type, act.region, 0,
                                   s.d->LocalDim(act.region), 2, 0.0f, 0);
                break;
            case MutationType::Expand2dTo3d:
                result.next = Expand2dTo3d(s, act.region, rng_());
                result.log.Record(act.type, act.region, 0,
                                   s.d->LocalDim(act.region), 3, 0.0f, 0);
                break;
            case MutationType::Collapse3dTo2d:
                result.next = Collapse3dTo2d(s, act.region);
                result.log.Record(act.type, act.region, 0,
                                   s.d->LocalDim(act.region), 2, 0.0f, 0);
                break;
            case MutationType::Collapse2dTo1d:
                result.next = Collapse2dTo1d(s, act.region);
                result.log.Record(act.type, act.region, 0,
                                   s.d->LocalDim(act.region), 1, 0.0f, 0);
                break;
            case MutationType::SplitRegion:
                result.next = ApplySplitRegion(s, act.region, act.split_axis);
                result.log.Record(act.type, act.region, 0, 0, 0, 0.0f, 0);
                break;
            case MutationType::LocalDimChange:
                result.next = ApplyLocalDimChange(s, act.region, act.target_dim, rng_());
                result.log.Record(act.type, act.region, 0,
                                   s.d->LocalDim(act.region), act.target_dim, 0.0f, 0);
                break;
            default:
                result.next    = s.Clone();
                result.mutated = false;
        }
    } catch (...) {
        result.next    = s.Clone();
        result.mutated = false;
    }

    return result;
}

std::unique_ptr<TopologicalOperator> MakeLearnedDtdo(int hidden_dim,
                                                       bool stochastic,
                                                       float temperature,
                                                       uint64_t seed) {
    return std::make_unique<DtdoNet>(
        hidden_dim,
        stochastic ? PolicyType::Stochastic : PolicyType::Greedy,
        temperature, seed);
}

} // namespace vbx
