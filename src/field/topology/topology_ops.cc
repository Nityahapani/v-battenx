#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include "vbatten_x/mutation_result.h"
#include <Eigen/Dense>
#include <algorithm>
#include <stdexcept>

namespace vbx {

FieldState SplitRegion(const FieldState& s, vbx_region_id rid, int axis) {
    FieldState out = s.Clone();

    auto params = s.F->Params();
    std::size_t half = params.size() / 2;

    std::vector<vbx_float> p_a(params.begin(), params.begin() + half);
    std::vector<vbx_float> p_b(params.begin() + half, params.end());

    out.F->Embed({p_a.data(), p_a.size()});

    vbx_region_id new_id = out.K->AddRegion();
    out.K->AddEdge(rid, new_id, 1.0f);

    vbx_dim_t dim = out.d->LocalDim(rid);
    out.d->SetDim(new_id, dim);

    return out;
}

FieldState MergeRegions(const FieldState& s, vbx_region_id r1, vbx_region_id r2) {
    auto neighbours = s.K->Neighbours(r1);
    bool adjacent = false;
    for (auto nb : neighbours) if (nb == r2) { adjacent = true; break; }
    if (!adjacent) throw std::runtime_error("MergeRegions: regions are not adjacent");

    FieldState out = s.Clone();
    auto p1 = s.F->Params();
    auto p2 = s.F->Params();

    std::vector<vbx_float> merged(p1.size());
    for (std::size_t i = 0; i < p1.size(); ++i)
        merged[i] = 0.5f * (p1[i] + p2[i]);

    out.F->Embed({merged.data(), merged.size()});
    out.K->RemoveEdge(r1, r2);
    out.K->RemoveRegion(r2);

    return out;
}

FieldState AddConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2) {
    FieldState out = s.Clone();
    out.K->AddEdge(r1, r2, 1.0f);
    return out;
}

FieldState RemoveConnection(const FieldState& s, vbx_region_id r1, vbx_region_id r2) {
    FieldState out = s.Clone();
    out.K->RemoveEdge(r1, r2);
    return out;
}

} // namespace vbx
