#include "vbatten_x/metric.h"
#include "src/field/topology/simplex_complex.cc"

namespace vbx {

struct TopologySnapshot {
    int regions;
    int edges;
    int betti0;
    int betti1;
    float diameter;
    float density;
};

TopologySnapshot ComputeTopologySnapshot(const FieldTopology& topo) {
    SimplexComplex sc{topo};
    return {
        static_cast<int>(topo.NumRegions()),
        static_cast<int>(topo.Edges().size()),
        sc.Betti0(),
        sc.Betti1(),
        static_cast<float>(sc.GraphDiameter()),
        sc.EdgeDensity()
    };
}

int TopologyDelta(const TopologySnapshot& before, const TopologySnapshot& after) {
    return std::abs(after.regions - before.regions)
         + std::abs(after.edges   - before.edges);
}

} // namespace vbx
