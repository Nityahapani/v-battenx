#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include "src/physics/pde/finite_diff_ops.cc"
#include <cmath>

namespace vbx {

class NavierStokesEvaluator : public PhysicsEvaluator {
public:
    NavierStokesEvaluator(float nu, float dt, int nx, int ny, float h)
        : nu_(nu), dt_(dt), nx_(nx), ny_(ny), h_(h) {}

    ResidualInfo Eval(const FieldState& state,
                      const PhysicalDataset&) const override {
        auto params = state.F->Params();
        int  n      = nx_ * ny_;
        int  n2     = 2 * n;

        GridField ux, uy;
        ux.nx = uy.nx = nx_;
        ux.ny = uy.ny = ny_;
        ux.h  = uy.h  = h_;
        ux.u.resize(n, 0.0f);
        uy.u.resize(n, 0.0f);

        for (int k = 0; k < n && k < static_cast<int>(params.size()); ++k)
            ux.u[k] = params[k];
        for (int k = 0; k < n && n+k < static_cast<int>(params.size()); ++k)
            uy.u[k] = params[n + k];

        double mom_res = 0.0, cont_res = 0.0;
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                float u = ux.at(i,j), v = uy.at(i,j);
                float lap_u = laplacian(ux, i, j);
                float lap_v = laplacian(uy, i, j);
                float conv_u = u * grad_x(ux,i,j) + v * grad_y(ux,i,j);
                float conv_v = u * grad_x(uy,i,j) + v * grad_y(uy,i,j);
                float ru = conv_u - nu_ * lap_u;
                float rv = conv_v - nu_ * lap_v;
                mom_res += ru*ru + rv*rv;

                float div = divergence(ux, uy, i, j);
                cont_res += div * div;
            }
        }
        float rms_mom  = static_cast<float>(std::sqrt(mom_res  / n));
        float rms_cont = static_cast<float>(std::sqrt(cont_res / n));
        return {{rms_mom}, {0.0f}, {rms_cont}};
    }

private:
    float nu_, dt_, h_;
    int   nx_, ny_;
};

std::unique_ptr<PhysicsEvaluator> MakeNavierStokesEvaluator(
    float nu, float dt, int nx, int ny, float h) {
    return std::make_unique<NavierStokesEvaluator>(nu, dt, nx, ny, h);
}

} // namespace vbx
