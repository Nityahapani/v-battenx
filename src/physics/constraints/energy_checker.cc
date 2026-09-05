#include "src/field/field_state.h"
#include <cmath>
#include <vector>

namespace vbx {

struct EnergyChecker {
    float threshold;
    float prev_energy = -1.0f;

    float energy(const FieldState& state) const {
        auto p = state.F->Params();
        double e = 0.0;
        for (float v : p) e += 0.5 * v * v;
        return static_cast<float>(e / std::max<std::size_t>(1, p.size()));
    }

    float check(const FieldState& state) {
        float e = energy(state);
        if (prev_energy < 0.0f) { prev_energy = e; return 0.0f; }
        float drift = std::abs(e - prev_energy) / (std::abs(prev_energy) + 1e-12f);
        prev_energy = e;
        return drift;
    }

    bool violated(const FieldState& state) {
        return check(state) > threshold;
    }
};

} // namespace vbx
