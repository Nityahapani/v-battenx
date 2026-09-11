#pragma once
#include <vector>
#include <cmath>

namespace vbx {

struct GridField {
    std::vector<float> u;
    int                nx;
    int                ny;
    float              h;

    float& at(int i, int j)              { return u[j * nx + i]; }
    float  at(int i, int j) const        { return u[j * nx + i]; }
    bool   in_bounds(int i, int j) const { return i >= 0 && i < nx && j >= 0 && j < ny; }

    // Ghost-cell value using zero-flux (Neumann ∂u/∂n=0) reflection.
    // Returns the mirror neighbour across the boundary so that all
    // central-difference stencils remain second-order O(h²) at boundaries.
    float ghost_x(int i, int j) const {
        if (i < 0)    return at(0,       j);
        if (i >= nx)  return at(nx - 1,  j);
        return at(i, j);
    }
    float ghost_y(int i, int j) const {
        if (j < 0)    return at(i, 0);
        if (j >= ny)  return at(i, ny - 1);
        return at(i, j);
    }
};

// Second-order central difference O(h²) everywhere via ghost cells.
inline float grad_x(const GridField& f, int i, int j) {
    return (f.ghost_x(i + 1, j) - f.ghost_x(i - 1, j)) / (2.0f * f.h);
}

inline float grad_y(const GridField& f, int i, int j) {
    return (f.ghost_y(i, j + 1) - f.ghost_y(i, j - 1)) / (2.0f * f.h);
}

// Five-point Laplacian with ghost-cell boundary treatment O(h²).
inline float laplacian(const GridField& f, int i, int j) {
    float xm = f.ghost_x(i - 1, j);
    float xp = f.ghost_x(i + 1, j);
    float ym = f.ghost_y(i, j - 1);
    float yp = f.ghost_y(i, j + 1);
    return (xm + xp + ym + yp - 4.0f * f.at(i, j)) / (f.h * f.h);
}

inline float divergence(const GridField& fx, const GridField& fy, int i, int j) {
    return grad_x(fx, i, j) + grad_y(fy, i, j);
}

inline float curl_2d(const GridField& fx, const GridField& fy, int i, int j) {
    return grad_x(fy, i, j) - grad_y(fx, i, j);
}

} // namespace vbx
