#include "src/physics/pde/grid_field.h"
#include <cmath>

namespace vbx {

float energy_flux_residual(const GridField& u_prev,
                            const GridField& u_curr,
                            float dt) {
    double e_prev = 0.0, e_curr = 0.0;
    for (float v : u_prev.u) e_prev += 0.5 * v * v;
    for (float v : u_curr.u) e_curr += 0.5 * v * v;
    e_prev /= u_prev.u.size();
    e_curr /= u_curr.u.size();
    return static_cast<float>(std::abs(e_curr - e_prev) / (std::abs(e_prev) + 1e-12));
}

} // namespace vbx
