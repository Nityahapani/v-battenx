#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include "src/physics/pde/grid_field.h"
#include <cmath>
#include <numeric>

namespace vbx {

class HeatEquationEvaluator : public PhysicsEvaluator {
public:
    explicit HeatEquationEvaluator(float alpha, float dt, int nx, int ny, float h)
        : alpha_(alpha), dt_(dt), nx_(nx), ny_(ny), h_(h) {}

    ResidualInfo Eval(const FieldState& state,
                      const PhysicalDataset& ds) const override {
        int         n      = nx_ * ny_;
        int         nrows  = static_cast<int>(ds.NumRows());
        int         ncols  = static_cast<int>(ds.NumCols());
        const float* raw   = ds.RawData();

        if (!raw || nrows == 0 || ncols < n)
            return {{0.0f}, {0.0f}, {0.0f}};

        auto params = state.F->Params();

        GridField u_curr, u_prev;
        u_curr.nx = u_prev.nx = nx_;
        u_curr.ny = u_prev.ny = ny_;
        u_curr.h  = u_prev.h  = h_;
        u_curr.u.resize(n, 0.0f);
        u_prev.u.resize(n, 0.0f);

        // u_curr: field state produced by the model (shared across rows).
        for (int k = 0; k < n && k < static_cast<int>(params.size()); ++k)
            u_curr.u[k] = params[k];

        double total = 0.0;
        int    count = 0;

        // Each dataset row is one flattened field snapshot u(x,t).
        // Residual: |(u_curr - u_row)/dt - α·∇²u_curr|² per grid cell per row.
        for (int r = 0; r < nrows; ++r) {
            for (int k = 0; k < n; ++k)
                u_prev.u[k] = raw[r * ncols + k];

            for (int j = 0; j < ny_; ++j) {
                for (int i = 0; i < nx_; ++i) {
                    float dudt  = (u_curr.at(i,j) - u_prev.at(i,j)) / dt_;
                    float lap   = laplacian(u_curr, i, j);
                    float resid = dudt - alpha_ * lap;
                    total += static_cast<double>(resid * resid);
                    ++count;
                }
            }
        }

        float rms = static_cast<float>(std::sqrt(total / std::max(count, 1)));
        return {{rms}, {0.0f}, {0.0f}};
    }

private:
    float alpha_, dt_, h_;
    int   nx_, ny_;
};

std::unique_ptr<PhysicsEvaluator> MakeHeatEvaluator(
    float alpha, float dt, int nx, int ny, float h) {
    return std::make_unique<HeatEquationEvaluator>(alpha, dt, nx, ny, h);
}

} // namespace vbx
