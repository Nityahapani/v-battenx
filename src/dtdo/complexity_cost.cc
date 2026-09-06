#include "vbatten_x/topological_operator.h"
#include "src/field/field_state.h"

namespace vbx {

ComplexityCost ComplexityCost::FromState(const FieldState& s,
                                          int max_dim, int max_reg, int max_con) {
    ComplexityCost c;
    c.max_total_dim   = max_dim;
    c.max_regions     = max_reg;
    c.max_connections = max_con;

    if (s.K) {
        c.current_regions     = static_cast<int>(s.K->NumRegions());
        c.current_connections = static_cast<int>(s.K->Edges().size());
    }
    if (s.d && s.K) {
        int total = 0;
        for (std::size_t r = 0; r < static_cast<std::size_t>(c.current_regions); ++r)
            total += s.d->LocalDim(static_cast<vbx_region_id>(r));
        c.current_total_dim = total;
    }
    return c;
}

} // namespace vbx
