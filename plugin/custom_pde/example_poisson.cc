#include "plugin/custom_pde/custom_pde_interface.h"
#include "src/field/field_state.h"
#include "src/physics/pde/grid_field.h"
#include <cmath>

namespace vbx {

class PoissonEvaluator : public PhysicsEvaluator {
public:
    PoissonEvaluator(int nx, int ny, float h) : nx_(nx), ny_(ny), h_(h) {}

    ResidualInfo Eval(const FieldState& state,
                       const PhysicalDataset& ds) const override {
        auto params = state.F->Params();
        int  n      = nx_ * ny_;

        GridField u;
        u.nx = nx_; u.ny = ny_; u.h = h_;
        u.u.resize(n, 0.0f);
        for (int k = 0; k < n && k < static_cast<int>(params.size()); ++k)
            u.u[k] = params[k];

        double total = 0.0;
        for (int j = 1; j < ny_ - 1; ++j)
            for (int i = 1; i < nx_ - 1; ++i) {
                float lap = laplacian(u, i, j);
                float f   = 1.0f;
                float r   = lap - f;
                total    += r * r;
            }

        float rms = static_cast<float>(std::sqrt(total / n));
        return {{rms}, {0.0f}, {0.0f}};
    }

private:
    int   nx_, ny_;
    float h_;
};

static std::unique_ptr<PhysicsEvaluator> MakePoisson() {
    return std::make_unique<PoissonEvaluator>(16, 16, 1.0f);
}

} // namespace vbx

VBATTENX_REGISTER_PDE(
    "poisson",
    vbx::MakePoisson,
    "Poisson equation ∇²u = f with f=1 on a 16×16 grid")
