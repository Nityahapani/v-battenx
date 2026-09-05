#include "vbatten_x/base.h"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace vbx {

struct GridField {
    std::vector<float> u;
    int                nx;
    int                ny;
    float              h;

    float& at(int i, int j)             { return u[j * nx + i]; }
    float  at(int i, int j) const       { return u[j * nx + i]; }
    bool   in_bounds(int i, int j) const { return i >= 0 && i < nx && j >= 0 && j < ny; }
};

float grad_x(const GridField& f, int i, int j) {
    int lo = std::max(0,      i - 1);
    int hi = std::min(f.nx-1, i + 1);
    return (f.at(hi, j) - f.at(lo, j)) / ((hi - lo) * f.h);
}

float grad_y(const GridField& f, int i, int j) {
    int lo = std::max(0,      j - 1);
    int hi = std::min(f.ny-1, j + 1);
    return (f.at(i, hi) - f.at(i, lo)) / ((hi - lo) * f.h);
}

float laplacian(const GridField& f, int i, int j) {
    float c  = f.at(i, j);
    float xm = (i > 0)      ? f.at(i-1, j) : c;
    float xp = (i < f.nx-1) ? f.at(i+1, j) : c;
    float ym = (j > 0)      ? f.at(i, j-1) : c;
    float yp = (j < f.ny-1) ? f.at(i, j+1) : c;
    return (xm + xp + ym + yp - 4.0f * c) / (f.h * f.h);
}

float divergence(const GridField& fx, const GridField& fy, int i, int j) {
    return grad_x(fx, i, j) + grad_y(fy, i, j);
}

float curl_2d(const GridField& fx, const GridField& fy, int i, int j) {
    return grad_x(fy, i, j) - grad_y(fx, i, j);
}

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
            if (mask[j * f.nx + i]) {
                float neighbor = 0.0f;
                int cnt = 0;
                if (i > 0      && !mask[j * f.nx + i-1]) { neighbor += f.at(i-1,j); ++cnt; }
                if (i < f.nx-1 && !mask[j * f.nx + i+1]) { neighbor += f.at(i+1,j); ++cnt; }
                if (j > 0      && !mask[(j-1)*f.nx + i]) { neighbor += f.at(i,j-1); ++cnt; }
                if (j < f.ny-1 && !mask[(j+1)*f.nx + i]) { neighbor += f.at(i,j+1); ++cnt; }
                if (cnt > 0) f.at(i,j) = neighbor / cnt + flux * f.h;
            }
}

void apply_periodic(GridField& f, int axis) {
    if (axis == 0) {
        for (int j = 0; j < f.ny; ++j) {
            f.at(0,       j) = f.at(f.nx-2, j);
            f.at(f.nx-1,  j) = f.at(1,      j);
        }
    } else {
        for (int i = 0; i < f.nx; ++i) {
            f.at(i, 0)      = f.at(i, f.ny-2);
            f.at(i, f.ny-1) = f.at(i, 1);
        }
    }
}

} // namespace vbx
