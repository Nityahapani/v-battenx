#include "vbatten_x/metric.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace vbx {

class RmseMetric : public PhysicsMetric {
public:
    vbx_float Eval(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        double s = 0.0;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            double d = pred[i] - label[i];
            s += d * d;
        }
        return static_cast<vbx_float>(std::sqrt(s / pred.size()));
    }
    std::string Name()     const override { return "rmse"; }
    bool        Maximize() const override { return false; }
};

class AucMetric : public PhysicsMetric {
public:
    vbx_float Eval(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        std::size_t n = pred.size();
        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            return pred[a] > pred[b];
        });

        double tp = 0, fp = 0, auc = 0;
        double pos = 0, neg = 0;
        for (std::size_t i = 0; i < n; ++i) pos += label[i];
        neg = n - pos;
        if (pos == 0 || neg == 0) return 0.5f;

        double prev_fp = 0, prev_tp = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (label[idx[i]] > 0.5f) tp++;
            else fp++;
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0 / (pos * neg);
            prev_fp = fp; prev_tp = tp;
        }
        return static_cast<vbx_float>(auc);
    }
    std::string Name()     const override { return "auc"; }
    bool        Maximize() const override { return true; }
};

std::unique_ptr<PhysicsMetric> MakeRmseMetric() { return std::make_unique<RmseMetric>(); }
std::unique_ptr<PhysicsMetric> MakeAucMetric()  { return std::make_unique<AucMetric>(); }

} // namespace vbx
