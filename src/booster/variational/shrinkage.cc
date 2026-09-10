#include "vbatten_x/base.h"
#include <cmath>

namespace vbx {

// ── Constant shrinkage (original behaviour) ──────────────────────────────────
class ShrinkageSchedule {
public:
    explicit ShrinkageSchedule(vbx_float lr) : lr_(lr) {}
    vbx_float operator()(int) const { return lr_; }
    void      SetLr(vbx_float lr)   { lr_ = lr; }
    vbx_float Lr() const            { return lr_; }

private:
    vbx_float lr_;
};

// ── Residual-Adaptive Shrinkage (RAS) ────────────────────────────────────────
//
// Motivation (trust-region shrinkage):
//   When the PDE residual r_t is large the field is in a high-curvature,
//   physics-violating region.  Taking a full step η₀ there risks overshooting
//   and destabilising subsequent mutations.  Conversely, when r_t ≈ 0 the
//   field is already physically consistent and a fuller step accelerates
//   convergence on the data-fit objective.
//
//   We therefore adapt the per-stage learning rate as:
//
//       η_t = η₀ / (1 + α · r_t)                                      (1)
//
//   where α ≥ 0 is a sensitivity parameter.  This satisfies:
//     • η_t = η₀  when r_t = 0   (full step in physics-consistent regions)
//     • η_t → 0   as r_t → ∞    (vanishing step in highly-violated regions)
//     • η_t is monotonically decreasing in r_t   ✓
//     • η_t is strictly positive for finite r_t  ✓
//
//   The schedule can be seen as a first-order Padé approximant to the
//   exponential dampening exp(-α·r_t), which is cheaper to evaluate and
//   avoids underflow when α·r_t is large.
//
//   Connection to Armijo–Goldstein:  The standard sufficient-decrease
//   condition requires the step size to satisfy
//       f(x - η·∇f) ≤ f(x) - c·η·‖∇f‖²
//   Under our physics-informed objective the effective gradient magnitude
//   scales with r_t (see physics_informed_obj.cc: phys_grad = 2·λ·r_t).
//   Choosing η_t ∝ 1/(1 + α·r_t) ensures the product η_t · ‖grad‖ remains
//   bounded, which is a necessary condition for Armijo satisfaction when
//   α ≈ λ_pde.
//
//   α = 0  recovers the original constant schedule exactly.
//
class ResidualAdaptiveShrinkage {
public:
    // lr     – base learning rate  η₀
    // alpha  – residual sensitivity  α  (recommended: 0.5–2.0)
    explicit ResidualAdaptiveShrinkage(vbx_float lr, vbx_float alpha = 1.0f)
        : lr_(lr), alpha_(alpha) {}

    // Per-stage call: update with the current PDE residual, then query.
    void     UpdateResidual(vbx_float pde_r) { last_r_ = pde_r; }
    vbx_float Compute() const {
        return lr_ / (1.0f + alpha_ * last_r_);
    }

    void      SetLr(vbx_float lr)    { lr_ = lr; }
    void      SetAlpha(vbx_float a)  { alpha_ = a; }
    vbx_float Lr()    const          { return lr_; }
    vbx_float Alpha() const          { return alpha_; }

private:
    vbx_float lr_;
    vbx_float alpha_;
    vbx_float last_r_ = 0.0f;
};

} // namespace vbx
