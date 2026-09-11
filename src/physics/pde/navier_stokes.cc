#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include "src/physics/pde/grid_field.h"
#include <cmath>

namespace vbx {

class NavierStokesEvaluator : public PhysicsEvaluator {
public:
    NavierStokesEvaluator(float nu, float dt, int nx, int ny, float h)
        : nu_(nu), dt_(dt), nx_(nx), ny_(ny), h_(h) {}

    // Incompressible NS: (u·∇)u = -∇p + ν∇²u,  ∇·u = 0.
    //
    // Without solving the pressure Poisson equation we cannot evaluate
    // ∇p directly. Instead we measure the *solenoidal* (divergence-free)
    // part of the momentum residual r = (u·∇)u - ν∇²u.
    //
    // Key identity: curl(∇p) = 0 in 2-D, so
    //   curl(r) = curl((u·∇)u - ν∇²u) = curl(-∇p + ν∇²u - ν∇²u + (u·∇)u)
    //           = curl((u·∇)u - ν∇²u)   [pressure term vanishes]
    //
    // Therefore ‖curl(r)‖ measures how far the velocity field is from
    // satisfying NS momentum without needing p.  Continuity residual
    // ‖∇·u‖ is unchanged.
    ResidualInfo Eval(const FieldState& state,
                      const PhysicalDataset&) const override {
        auto params = state.F->Params();
        int  n      = nx_ * ny_;

        GridField ux, uy;
        ux.nx = uy.nx = nx_;
        ux.ny = uy.ny = ny_;
        ux.h  = uy.h  = h_;
        ux.u.resize(n, 0.0f);
        uy.u.resize(n, 0.0f);

        for (int k = 0; k < n && k < static_cast<int>(params.size()); ++k)
            ux.u[k] = params[k];
        for (int k = 0; k < n && n + k < static_cast<int>(params.size()); ++k)
            uy.u[k] = params[n + k];

        // Build residual vector fields rx, ry = (u·∇)u - ν∇²u.
        GridField rx, ry;
        rx.nx = ry.nx = nx_;
        rx.ny = ry.ny = ny_;
        rx.h  = ry.h  = h_;
        rx.u.resize(n, 0.0f);
        ry.u.resize(n, 0.0f);

        double cont_res = 0.0;
        for (int j = 0; j < ny_; ++j) {
            for (int i = 0; i < nx_; ++i) {
                float u = ux.at(i,j), v = uy.at(i,j);
                rx.at(i,j) = u * grad_x(ux,i,j) + v * grad_y(ux,i,j) - nu_ * laplacian(ux,i,j);
                ry.at(i,j) = u * grad_x(uy,i,j) + v * grad_y(uy,i,j) - nu_ * laplacian(uy,i,j);
                float div   = divergence(ux, uy, i, j);
                cont_res   += div * div;
            }
        }

        // Solenoidal momentum residual: ‖curl(r)‖ is pressure-free.
        double mom_res = 0.0;
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i) {
                float c = curl_2d(rx, ry, i, j);
                mom_res += c * c;
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
