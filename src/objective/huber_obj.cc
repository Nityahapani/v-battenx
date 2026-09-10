#include "vbatten_x/objective.h"
#include <cmath>
#include <algorithm>

namespace vbx {

class HuberObjective : public Objective {
public:
    explicit HuberObjective(vbx_float delta = 1.0f) : delta_(delta) {}

    GradPair GetGradients(Span<const vbx_float> pred,
                          Span<const vbx_float> label) const override {
        GradPair gp;
        gp.g.resize(pred.size());
        gp.h.resize(pred.size());
        for (std::size_t i = 0; i < pred.size(); ++i) {
            vbx_float r = pred[i] - label[i];
            vbx_float abs_r = std::abs(r);
            if (abs_r <= delta_) {
                gp.g[i] = r;
                gp.h[i] = 1.0f;
            } else {
                gp.g[i] = delta_ * (r > 0.0f ? 1.0f : -1.0f);
                gp.h[i] = delta_ / abs_r;
            }
        }
        return gp;
    }

    vbx_float Loss(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        double s = 0.0;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            double r = pred[i] - label[i];
            double abs_r = std::abs(r);
            s += abs_r <= delta_ ? 0.5 * r * r
                                 : delta_ * (abs_r - 0.5 * delta_);
        }
        return static_cast<vbx_float>(s / pred.size());
    }

    std::string Name() const override { return "huber"; }

private:
    vbx_float delta_;
};

std::unique_ptr<Objective> MakeHuberObjective(vbx_float delta) {
    return std::make_unique<HuberObjective>(delta);
}

} // namespace vbx
