#include "src/field/topology/simplex_complex.h"

namespace vbx {

float BettiCost(const FieldTopology& topo) {
    SimplexComplex sc{topo};
    return static_cast<float>(sc.Betti1());
}

float DiameterCost(const FieldTopology& topo) {
    SimplexComplex sc{topo};
    return static_cast<float>(sc.GraphDiameter());
}

float DensityCost(const FieldTopology& topo) {
    SimplexComplex sc{topo};
    return sc.EdgeDensity();
}

float TotalTopologyCost(const FieldTopology& topo) {
    return BettiCost(topo) + 0.5f * DiameterCost(topo) + DensityCost(topo);
}

} // namespace vbx
