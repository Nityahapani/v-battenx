#include "vbatten_x/base.h"

namespace vbx {

class ShrinkageSchedule {
public:
    explicit ShrinkageSchedule(vbx_float lr) : lr_(lr) {}

    vbx_float operator()(int /*stage*/) const { return lr_; }

    void SetLr(vbx_float lr) { lr_ = lr; }
    vbx_float Lr() const { return lr_; }

private:
    vbx_float lr_;
};

} // namespace vbx
