#include "field_state.h"
#include <cmath>
#include <numeric>

namespace vbx {

FieldState FieldState::Clone() const {
    FieldState s;
    if (F) s.F = F->Clone();
    if (K) s.K = K->Clone();
    if (d) s.d = d->Clone();
    if (T) s.T = T->Clone();
    s.bias = bias;
    return s;
}

double FieldState::Diff(const FieldState& other) const {
    if (!F || !other.F) return 0.0;
    auto a = F->Params();
    auto b = other.F->Params();
    double sq = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        double diff = a[i] - b[i];
        sq += diff * diff;
    }
    return std::sqrt(sq);
}

} // namespace vbx
