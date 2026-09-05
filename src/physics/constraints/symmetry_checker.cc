#include "src/field/field_state.h"
#include "vbatten_x/data.h"
#include <cmath>
#include <vector>

namespace vbx {

float CheckRotation2dEquivariance(const FieldState& state, int num_samples = 8) {
    auto params = state.F->Params();
    std::size_t dim = params.size();
    if (dim < 2) return 0.0f;

    double total = 0.0;
    float  pi    = 3.14159265f;
    for (int k = 1; k < num_samples; ++k) {
        float  theta = 2.0f * pi * k / num_samples;
        float  c     = std::cos(theta);
        float  s     = std::sin(theta);
        double sq    = 0.0;
        for (std::size_t i = 0; i + 1 < dim; i += 2) {
            float x  = params[i];
            float y  = params[i+1];
            float rx = c * x - s * y;
            float ry = s * x + c * y;
            float dx = rx - x;
            float dy = ry - y;
            sq += dx*dx + dy*dy;
        }
        total += std::sqrt(sq / (dim / 2));
    }
    return static_cast<float>(total / (num_samples - 1));
}

float CheckReflectionEquivariance(const FieldState& state) {
    auto params = state.F->Params();
    std::size_t dim = params.size();
    double sq = 0.0;
    for (std::size_t i = 0; i + 1 < dim; i += 2) {
        float orig = params[i];
        float refl = -params[i];
        float diff = refl - orig;
        sq += diff * diff;
    }
    return static_cast<float>(std::sqrt(sq / std::max<std::size_t>(1, dim / 2)));
}

} // namespace vbx
