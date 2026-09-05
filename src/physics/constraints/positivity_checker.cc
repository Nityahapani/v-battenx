#include "src/field/field_state.h"
#include <cstddef>

namespace vbx {

struct PositivityResult {
    int   num_violations;
    float min_value;
};

PositivityResult CheckPositivity(const FieldState& state) {
    auto p = state.F->Params();
    int   nviol = 0;
    float minv  = p.empty() ? 0.0f : p[0];
    for (float v : p) {
        if (v < 0.0f) ++nviol;
        if (v < minv)  minv = v;
    }
    return {nviol, minv};
}

} // namespace vbx
