#include "vbatten_x/topological_operator.h"
#include "src/dtdo/mutation_router.cc"
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace vbx {

std::unique_ptr<TopologicalOperator> MakeOperator(const std::string& name,
                                                    float tau_expand   = 0.1f,
                                                    float tau_collapse = 0.01f) {
    if (name == "threshold")
        return MakeThresholdOperator(tau_expand, tau_collapse);
    if (name == "gradient")
        return MakeGradientOperator();
    if (name == "pruner")
        return MakeComplexityPruner();
    if (name == "router" || name == "default")
        return MakeMutationRouter(tau_expand, tau_collapse);
    if (name == "none" || name == "")
        return nullptr;
    throw std::runtime_error("Unknown TopologicalOperator: " + name);
}

} // namespace vbx
