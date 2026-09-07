#include "src/physics/pde/finite_diff_ops.cc"
#include "src/common/timer.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <numeric>

using namespace vbx;

static GridField SynthField(int nx, int ny) {
    GridField f;
    f.nx = nx; f.ny = ny; f.h = 1.0f / nx;
    f.u.resize(nx * ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            f.u[j * nx + i] = std::sin(3.14159f * i / nx)
                             * std::sin(3.14159f * j / ny);
    return f;
}

int main() {
    std::vector<int> sizes = {16, 32, 64, 128, 256};
    int reps               = 20;

    std::cout << std::left << std::setw(10) << "Grid"
              << std::setw(14) << "Time(ms)"
              << std::setw(14) << "ms/laplacian"
              << "Max|∇²u + 2π²u|\n";
    std::cout << std::string(60, '-') << "\n";

    for (int sz : sizes) {
        GridField f = SynthField(sz, sz);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; ++r)
            for (int j = 1; j < sz-1; ++j)
                for (int i = 1; i < sz-1; ++i)
                    laplacian(f, i, j);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Verify: ∇²(sin πx/L sin πy/L) = -2(π/L)² sin πx/L sin πy/L
        float max_err = 0.0f;
        float pi_sq   = 2.0f * (3.14159f / sz) * (3.14159f / sz);
        for (int j = 1; j < sz-1; ++j)
            for (int i = 1; i < sz-1; ++i) {
                float lap   = laplacian(f, i, j);
                float exact = -pi_sq * f.at(i, j) * sz * sz;
                max_err = std::max(max_err, std::abs(lap - exact / (sz * sz)));
            }

        std::cout << std::left << std::setw(10) << (std::to_string(sz) + "x" + std::to_string(sz))
                  << std::setw(14) << std::fixed << std::setprecision(2) << ms
                  << std::setw(14) << std::setprecision(3) << ms / reps
                  << std::setprecision(5) << max_err << "\n";
    }
    return 0;
}
