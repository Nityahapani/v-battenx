#include "src/physics/pde/grid_field.h"
#include <cmath>
#include <vector>

namespace vbx {

float divergence_free_residual(const GridField& fx, const GridField& fy) {
    double sum = 0.0;
    for (int j = 0; j < fy.ny; ++j)
        for (int i = 0; i < fx.nx; ++i) {
            float d = divergence(fx, fy, i, j);
            sum += d * d;
        }
    return static_cast<float>(std::sqrt(sum / (fx.nx * fx.ny)));
}

float curl_free_residual(const GridField& fx, const GridField& fy) {
    double sum = 0.0;
    for (int j = 0; j < fy.ny; ++j)
        for (int i = 0; i < fx.nx; ++i) {
            float c = curl_2d(fx, fy, i, j);
            sum += c * c;
        }
    return static_cast<float>(std::sqrt(sum / (fx.nx * fx.ny)));
}

void apply_dirichlet(GridField& f, const std::vector<bool>& mask, float value) {
    for (std::size_t k = 0; k < f.u.size(); ++k)
        if (mask[k]) f.u[k] = value;
}

void apply_neumann(GridField& f, const std::vector<bool>& mask, float flux) {
    for (int j = 0; j < f.ny; ++j)
        for (int i = 0; i < f.nx; ++i)
            if (mask[static_cast<std::size_t>(j * f.nx + i)]) {
                float neighbor = 0.0f; int cnt = 0;
                if (i > 0      && !mask[static_cast<std::size_t>(j*f.nx+i-1)]) { neighbor += f.at(i-1,j); ++cnt; }
                if (i < f.nx-1 && !mask[static_cast<std::size_t>(j*f.nx+i+1)]) { neighbor += f.at(i+1,j); ++cnt; }
                if (j > 0      && !mask[static_cast<std::size_t>((j-1)*f.nx+i)]) { neighbor += f.at(i,j-1); ++cnt; }
                if (j < f.ny-1 && !mask[static_cast<std::size_t>((j+1)*f.nx+i)]) { neighbor += f.at(i,j+1); ++cnt; }
                if (cnt > 0) f.at(i,j) = neighbor / cnt + flux * f.h;
            }
}

void apply_periodic(GridField& f, int axis) {
    if (axis == 0) {
        for (int j = 0; j < f.ny; ++j) {
            f.at(0,      j) = f.at(f.nx-2, j);
            f.at(f.nx-1, j) = f.at(1,      j);
        }
    } else {
        for (int i = 0; i < f.nx; ++i) {
            f.at(i, 0)      = f.at(i, f.ny-2);
            f.at(i, f.ny-1) = f.at(i, 1);
        }
    }
}

} // namespace vbx
