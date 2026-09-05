#include "vbatten_x/objective.h"
#include <cmath>
#include <algorithm>

namespace vbx {

static inline vbx_float Sigmoid(vbx_float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

class BinaryLogisticObjective : public Objective {
public:
    GradPair GetGradients(Span<const vbx_float> pred,
                          Span<const vbx_float> label) const override {
        GradPair gp;
        gp.g.resize(pred.size());
        gp.h.resize(pred.size());
        for (std::size_t i = 0; i < pred.size(); ++i) {
            vbx_float p = Sigmoid(pred[i]);
            gp.g[i] = p - label[i];
            gp.h[i] = std::max(p * (1.0f - p), 1e-7f);
        }
        return gp;
    }

    vbx_float Loss(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        double s = 0.0;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            vbx_float p = Sigmoid(pred[i]);
            p = std::max(1e-7f, std::min(1.0f - 1e-7f, p));
            s -= label[i] * std::log(p) + (1.0f - label[i]) * std::log(1.0f - p);
        }
        return static_cast<vbx_float>(s / pred.size());
    }

    std::string Name() const override { return "binary_logistic"; }
};

std::unique_ptr<Objective> MakeClassificationObjective() {
    return std::make_unique<BinaryLogisticObjective>();
}

} // namespace vbx
