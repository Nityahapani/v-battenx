#include "vbatten_x/base.h"

namespace vbx {

// Residual-Adaptive Shrinkage: lr_t = lr_0 / (1 + α·‖g_t‖)
// where ‖g_t‖ is the L2 norm of the boosting gradient at step t.
// α controls sensitivity: α=0 reduces to constant LR, α→∞ gives aggressive decay.
class ShrinkageSchedule {
public:
    ShrinkageSchedule(vbx_float lr, vbx_float alpha = 1.0f)
        : lr0_(lr), alpha_(alpha), current_lr_(lr) {}

    void Update(vbx_float grad_norm) {
        current_lr_ = lr0_ / (1.0f + alpha_ * grad_norm);
    }

    vbx_float operator()(int) const { return current_lr_; }
    void      SetLr(vbx_float lr)   { lr0_ = lr; current_lr_ = lr; }
    vbx_float Lr() const            { return current_lr_; }
    vbx_float BaseLr() const        { return lr0_; }

private:
    vbx_float lr0_;
    vbx_float alpha_;
    vbx_float current_lr_;
};

} // namespace vbx
