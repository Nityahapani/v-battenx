#include "vbatten_x/metric.h"
#include "vbatten_x/physics_evaluator.h"
#include <cmath>
#include <algorithm>

namespace vbx {

class PdeResidualL2Metric : public PhysicsMetric {
public:
    vbx_float Eval(Span<const vbx_float> residuals,
                   Span<const vbx_float>) const override {
        double sq = 0.0;
        for (auto r : residuals) sq += r * r;
        return static_cast<vbx_float>(std::sqrt(sq / std::max<std::size_t>(1, residuals.size())));
    }
    std::string Name()     const override { return "pde_residual_l2"; }
    bool        Maximize() const override { return false; }
};

class PdeResidualLinfMetric : public PhysicsMetric {
public:
    vbx_float Eval(Span<const vbx_float> residuals,
                   Span<const vbx_float>) const override {
        float mx = 0.0f;
        for (auto r : residuals) mx = std::max(mx, std::abs(r));
        return mx;
    }
    std::string Name()     const override { return "pde_residual_linf"; }
    bool        Maximize() const override { return false; }
};

class SymmetryViolationMetric : public PhysicsMetric {
public:
    vbx_float Eval(Span<const vbx_float> violations,
                   Span<const vbx_float>) const override {
        if (violations.empty()) return 0.0f;
        double s = 0.0;
        for (auto v : violations) s += v;
        return static_cast<vbx_float>(s / violations.size());
    }
    std::string Name()     const override { return "symmetry_violation"; }
    bool        Maximize() const override { return false; }
};

std::unique_ptr<PhysicsMetric> MakePdeL2Metric()           { return std::make_unique<PdeResidualL2Metric>(); }
std::unique_ptr<PhysicsMetric> MakePdeLinfMetric()         { return std::make_unique<PdeResidualLinfMetric>(); }
std::unique_ptr<PhysicsMetric> MakeSymmetryViolationMetric(){ return std::make_unique<SymmetryViolationMetric>(); }

} // namespace vbx
