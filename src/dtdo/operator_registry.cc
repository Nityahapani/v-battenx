#include "vbatten_x/topological_operator.h"
#include "src/dtdo/mutation_router.cc"
#include "src/dtdo/learned/dtdo_net_apply.cc"
#include <string>
#include <stdexcept>

namespace vbx {

std::unique_ptr<TopologicalOperator> MakeOperator(const std::string& name,
                                                    float tau_expand,
                                                    float tau_collapse) {
    if (name == "threshold")
        return MakeThresholdOperator(tau_expand, tau_collapse);
    if (name == "gradient")
        return MakeGradientOperator(0.05f, 0.005f);
    if (name == "pruner")
        return MakeComplexityPruner(5);
    if (name == "router" || name == "default")
        return MakeMutationRouter(tau_expand, tau_collapse, 0.05f, 0.005f, 5);
    if (name == "learned")
        return MakeLearnedDtdo(64, true, 1.0f, 42);
    if (name == "learned_greedy")
        return MakeLearnedDtdo(64, false, 1.0f, 42);
    if (name == "none" || name.empty())
        return nullptr;
    throw std::runtime_error("Unknown TopologicalOperator: " + name);
}

} // namespace vbx
