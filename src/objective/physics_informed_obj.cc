#include "vbatten_x/objective.h"
#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include <memory>
#include <cmath>

namespace vbx {

class PhysicsInformedObjective : public Objective {
public:
    PhysicsInformedObjective(std::unique_ptr<Objective>        task_obj,
                              float                             lambda_physics)
        : task_(std::move(task_obj))
        , lambda_(lambda_physics) {}

    GradPair GetGradients(Span<const vbx_float> pred,
                          Span<const vbx_float> label) const override {
        auto gp = task_->GetGradients(pred, label);
        float phys_grad = 2.0f * lambda_ * last_pde_residual_;
        for (auto& v : gp.g) v += phys_grad;
        return gp;
    }

    vbx_float Loss(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        return task_->Loss(pred, label)
               + lambda_ * last_pde_residual_ * last_pde_residual_;
    }

    void UpdatePdeResidual(float r) { last_pde_residual_ = r; }

    std::string Name() const override { return "physics_informed"; }

private:
    std::unique_ptr<Objective> task_;
    float  lambda_;
    mutable float last_pde_residual_ = 0.0f;
};

} // namespace vbx
