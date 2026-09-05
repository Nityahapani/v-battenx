#include "vbatten_x/objective.h"
#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include <memory>
#include <cmath>

namespace vbx {

class PdeConstrainedObjective : public Objective {
public:
    PdeConstrainedObjective(std::unique_ptr<Objective>        task_obj,
                             std::unique_ptr<PhysicsEvaluator> evaluator,
                             float lambda_pde,
                             float lambda_bc)
        : task_(std::move(task_obj))
        , evaluator_(std::move(evaluator))
        , lambda_pde_(lambda_pde)
        , lambda_bc_(lambda_bc) {}

    GradPair GetGradients(Span<const vbx_float> pred,
                          Span<const vbx_float> label) const override {
        auto gp = task_->GetGradients(pred, label);
        if (last_pde_residual_ > 0.0f) {
            float pde_grad_scale = lambda_pde_ * last_pde_residual_;
            for (auto& v : gp.g) v += pde_grad_scale;
        }
        return gp;
    }

    vbx_float Loss(Span<const vbx_float> pred,
                   Span<const vbx_float> label) const override {
        float task_loss = task_->Loss(pred, label);
        float pde_term  = lambda_pde_ * last_pde_residual_ * last_pde_residual_;
        float bc_term   = lambda_bc_  * last_bc_residual_  * last_bc_residual_;
        return task_loss + pde_term + bc_term;
    }

    void UpdateResiduals(float pde_r, float bc_r) {
        last_pde_residual_ = pde_r;
        last_bc_residual_  = bc_r;
    }

    std::string Name() const override { return "pde_constrained"; }

private:
    std::unique_ptr<Objective>        task_;
    std::unique_ptr<PhysicsEvaluator> evaluator_;
    float  lambda_pde_;
    float  lambda_bc_;
    mutable float last_pde_residual_ = 0.0f;
    mutable float last_bc_residual_  = 0.0f;
};

} // namespace vbx
