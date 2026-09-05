#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

namespace vbx {

struct GridField {
    std::vector<float> u;
    int                nx;
    int                ny;
    float              h;

    float& at(int i, int j)              { return u[j * nx + i]; }
    float  at(int i, int j) const        { return u[j * nx + i]; }
    bool   in_bounds(int i, int j) const { return i >= 0 && i < nx && j >= 0 && j < ny; }
};

inline float grad_x(const GridField& f, int i, int j) {
    int lo = std::max(0,      i - 1);
    int hi = std::min(f.nx-1, i + 1);
    return (f.at(hi, j) - f.at(lo, j)) / ((hi - lo) * f.h);
}

inline float grad_y(const GridField& f, int i, int j) {
    int lo = std::max(0,      j - 1);
    int hi = std::min(f.ny-1, j + 1);
    return (f.at(i, hi) - f.at(i, lo)) / ((hi - lo) * f.h);
}

inline float laplacian(const GridField& f, int i, int j) {
    float c  = f.at(i, j);
    float xm = (i > 0)      ? f.at(i-1, j) : c;
    float xp = (i < f.nx-1) ? f.at(i+1, j) : c;
    float ym = (j > 0)      ? f.at(i, j-1) : c;
    float yp = (j < f.ny-1) ? f.at(i, j+1) : c;
    return (xm + xp + ym + yp - 4.0f * c) / (f.h * f.h);
}

inline float divergence(const GridField& fx, const GridField& fy, int i, int j) {
    return grad_x(fx, i, j) + grad_y(fy, i, j);
}

inline float curl_2d(const GridField& fx, const GridField& fy, int i, int j) {
    return grad_x(fy, i, j) - grad_y(fx, i, j);
}

} // namespace vbx
