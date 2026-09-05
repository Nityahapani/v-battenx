#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include <functional>

namespace vbx {

using PdeResidualFn = std::function<
    float(const std::vector<float>& params,
          const PhysicalDataset&    ds)>;

class CustomPdeEvaluator : public PhysicsEvaluator {
public:
    explicit CustomPdeEvaluator(PdeResidualFn fn) : fn_(std::move(fn)) {}

    ResidualInfo Eval(const FieldState& state,
                      const PhysicalDataset& ds) const override {
        auto params = state.F->Params();
        float r = fn_(params, ds);
        return {{r}, {0.0f}, {0.0f}};
    }

private:
    PdeResidualFn fn_;
};

std::unique_ptr<PhysicsEvaluator> MakeCustomPdeEvaluator(PdeResidualFn fn) {
    return std::make_unique<CustomPdeEvaluator>(std::move(fn));
}

} // namespace vbx
