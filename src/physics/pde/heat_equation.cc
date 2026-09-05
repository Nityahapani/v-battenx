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
        auto params = state.F->Params();
        int  n      = nx_ * ny_;

        GridField u_curr;
        u_curr.nx = nx_;
        u_curr.ny = ny_;
        u_curr.h  = h_;
        u_curr.u.resize(n);
        for (int k = 0; k < n && k < static_cast<int>(params.size()); ++k)
            u_curr.u[k] = params[k];

        GridField u_prev = u_curr;
        auto raw = ds.RawData();
        if (raw && ds.NumRows() >= 1 && ds.NumCols() >= n) {
            for (int k = 0; k < n; ++k)
                u_prev.u[k] = raw[k];
        }

        double total = 0.0;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                float dudt  = (u_curr.at(i,j) - u_prev.at(i,j)) / dt_;
                float lap   = laplacian(u_curr, i, j);
                float resid = dudt - alpha_ * lap;
                total += resid * resid;
            }

        float rms = static_cast<float>(std::sqrt(total / n));
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
