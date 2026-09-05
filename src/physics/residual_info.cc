#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include <cmath>

namespace vbx {

vbx_float ResidualInfo::MeanPde() const {
    if (pde_residual.empty()) return 0.0f;
    double s = 0.0;
    for (auto v : pde_residual) s += std::abs(static_cast<double>(v));
    return static_cast<vbx_float>(s / pde_residual.size());
}

vbx_float ResidualInfo::MeanPrediction() const {
    if (prediction_residual.empty()) return 0.0f;
    double s = 0.0;
    for (auto v : prediction_residual) s += std::abs(static_cast<double>(v));
    return static_cast<vbx_float>(s / prediction_residual.size());
}

class NullEvaluator : public PhysicsEvaluator {
public:
    ResidualInfo Eval(const FieldState&, const PhysicalDataset&) const override {
        return {{0.0f}, {0.0f}, {0.0f}};
    }
};

std::unique_ptr<PhysicsEvaluator> MakeNullEvaluator() {
    return std::make_unique<NullEvaluator>();
}

} // namespace vbx
