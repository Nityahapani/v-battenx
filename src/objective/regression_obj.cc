#include "vbatten_x/objective.h"
#include <cmath>

namespace vbx {

class RegressionObjective : public Objective {
public:
    GradPair GetGradients(Span<const vbx_float> pred,
                          Span<const vbx_float> label) const override {
        GradPair gp;
        gp.g.resize(pred.size());
        gp.h.resize(pred.size(), 1.0f);
        for (std::size_t i = 0; i < pred.size(); ++i)
            gp.g[i] = pred[i] - label[i];
        return gp;
    }

    vbx_float Loss(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        double s = 0.0;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            double d = pred[i] - label[i];
            s += d * d;
        }
        return static_cast<vbx_float>(s / pred.size());
    }

    std::string Name() const override { return "mse"; }
};

std::unique_ptr<Objective> MakeRegressionObjective() {
    return std::make_unique<RegressionObjective>();
}

} // namespace vbx
