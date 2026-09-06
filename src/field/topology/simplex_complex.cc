#include "vbatten_x/topology.h"
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <cmath>

namespace vbx {

struct SimplexComplex {
    const FieldTopology& topo;

    int Betti0() const {
        std::size_t n = topo.NumRegions();
        if (n == 0) return 0;
        std::unordered_set<vbx_region_id> visited;
        std::queue<vbx_region_id> q;
        auto edges = topo.Edges();
        int components = 0;

        std::unordered_set<vbx_region_id> all_nodes;
        for (auto& e : edges) { all_nodes.insert(e.src); all_nodes.insert(e.dst); }

        for (auto node : all_nodes) {
            if (!visited.count(node)) {
                ++components;
                q.push(node);
                while (!q.empty()) {
                    auto r = q.front(); q.pop();
                    if (!visited.insert(r).second) continue;
                    for (auto nb : topo.Neighbours(r)) q.push(nb);
                }
            }
        }
        return std::max(1, components);
    }

    int Betti1() const {
        auto edges  = topo.Edges();
        int  V      = static_cast<int>(topo.NumRegions());
        int  E      = static_cast<int>(edges.size());
        int  b0     = Betti0();
        return std::max(0, E - V + b0);
    }

    int GraphDiameter() const {
        auto edges = topo.Edges();
        if (edges.empty()) return 0;
        std::unordered_set<vbx_region_id> nodes;
        for (auto& e : edges) { nodes.insert(e.src); nodes.insert(e.dst); }

        int diameter = 0;
        for (auto src : nodes) {
            std::unordered_map<vbx_region_id, int> dist;
            std::queue<vbx_region_id> q;
            dist[src] = 0; q.push(src);
            while (!q.empty()) {
                auto r = q.front(); q.pop();
                for (auto nb : topo.Neighbours(r)) {
                    if (!dist.count(nb)) {
                        dist[nb] = dist[r] + 1;
                        diameter = std::max(diameter, dist[nb]);
                        q.push(nb);
                    }
                }
            }
        }
        return diameter;
    }

    float EdgeDensity() const {
        int V = static_cast<int>(topo.NumRegions());
        int E = static_cast<int>(topo.Edges().size());
        if (V < 2) return 0.0f;
        int max_edges = V * (V - 1) / 2;
        return static_cast<float>(E) / max_edges;
    }
};

} // namespace vbx
