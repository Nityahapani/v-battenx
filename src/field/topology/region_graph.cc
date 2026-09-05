#include "vbatten_x/topology.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>

namespace vbx {

class RegionGraph : public FieldTopology {
public:
    RegionGraph() { AddRegion(); }

    std::size_t NumRegions() const override { return adj_.size(); }

    std::vector<vbx_region_id> Neighbours(vbx_region_id r) const override {
        auto it = adj_.find(r);
        if (it == adj_.end()) return {};
        std::vector<vbx_region_id> out;
        for (auto& [nb, _] : it->second) out.push_back(nb);
        return out;
    }

    std::vector<RegionEdge> Edges() const override {
        std::vector<RegionEdge> out;
        for (auto& [src, nbmap] : adj_)
            for (auto& [dst, w] : nbmap)
                if (src <= dst) out.push_back({src, dst, w});
        return out;
    }

    bool IsConnected() const override {
        if (adj_.empty()) return true;
        std::unordered_set<vbx_region_id> visited;
        std::queue<vbx_region_id> q;
        q.push(adj_.begin()->first);
        while (!q.empty()) {
            auto r = q.front(); q.pop();
            if (visited.count(r)) continue;
            visited.insert(r);
            for (auto nb : Neighbours(r)) q.push(nb);
        }
        return visited.size() == adj_.size();
    }

    vbx_region_id AddRegion() override {
        vbx_region_id id = next_id_++;
        adj_[id] = {};
        return id;
    }

    void RemoveRegion(vbx_region_id r) override {
        adj_.erase(r);
        for (auto& [_, nbmap] : adj_) nbmap.erase(r);
    }

    void AddEdge(vbx_region_id a, vbx_region_id b, vbx_float w) override {
        adj_[a][b] = w;
        adj_[b][a] = w;
    }

    void RemoveEdge(vbx_region_id a, vbx_region_id b) override {
        adj_[a].erase(b);
        adj_[b].erase(a);
    }

    std::unique_ptr<FieldTopology> Clone() const override {
        auto c = std::make_unique<RegionGraph>();
        c->adj_     = adj_;
        c->next_id_ = next_id_;
        return c;
    }

private:
    std::unordered_map<vbx_region_id,
        std::unordered_map<vbx_region_id, vbx_float>> adj_;
    vbx_region_id next_id_ = 0;
};

std::unique_ptr<FieldTopology> MakeRegionGraph() {
    return std::make_unique<RegionGraph>();
}

} // namespace vbx
