#include "vbatten_x/base.h"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace vbx {

class AdamOptimizer {
public:
    AdamOptimizer(float lr = 1e-3f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float eps = 1e-8f,
                  float grad_clip = 0.0f)
        : lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps),
          grad_clip_(grad_clip), t_(0) {}

    void Step(Eigen::VectorXf& params, const Eigen::VectorXf& grad) {
        if (m_.size() != params.size()) {
            m_ = Eigen::VectorXf::Zero(params.size());
            v_ = Eigen::VectorXf::Zero(params.size());
        }

        Eigen::VectorXf g = grad;
        if (grad_clip_ > 0.0f) {
            float gnorm = g.norm();
            if (gnorm > grad_clip_) g *= grad_clip_ / gnorm;
        }

        ++t_;
        m_ = beta1_ * m_ + (1.0f - beta1_) * g;
        v_ = beta2_ * v_ + (1.0f - beta2_) * g.cwiseProduct(g);

        float bc1 = 1.0f - std::pow(beta1_, t_);
        float bc2 = 1.0f - std::pow(beta2_, t_);
        Eigen::VectorXf m_hat = m_ / bc1;
        Eigen::VectorXf v_hat = v_ / bc2;

        params -= lr_ * m_hat.array() / (v_hat.array().sqrt() + eps_);
    }

    void Reset() { m_.setZero(); v_.setZero(); t_ = 0; }

    void SetLr(float lr) { lr_ = lr; }
    float Lr() const     { return lr_; }
    int   Steps() const  { return t_; }

private:
    float           lr_, beta1_, beta2_, eps_, grad_clip_;
    int             t_;
    Eigen::VectorXf m_, v_;
};

class LbfgsOptimizer {
public:
    explicit LbfgsOptimizer(int memory = 10, float lr = 1.0f)
        : m_(memory), lr_(lr) {}

    // grad must be ∇f at the current params.
    // On return, params is updated to θ_{k+1} and the curvature pair
    // (s_k = θ_{k+1} - θ_k,  y_k = ∇f(θ_{k+1}) - ∇f(θ_k)) is stored
    // on the *next* call once the caller re-evaluates the gradient.
    void Step(Eigen::VectorXf& params, const Eigen::VectorXf& grad) {
        // Build two-loop recursion with curvature pairs accumulated so far.
        Eigen::VectorXf q = grad;
        int k = static_cast<int>(s_list_.size());
        std::vector<float> alpha(k);

        for (int i = k - 1; i >= 0; --i) {
            alpha[i] = rho_list_[i] * s_list_[i].dot(q);
            q -= alpha[i] * y_list_[i];
        }

        Eigen::VectorXf r = q;
        if (k > 0) {
            // Nocedal & Wright (2006) eq. 7.20: γ = sᵀy / yᵀy
            float gamma = s_list_.back().dot(y_list_.back())
                        / (y_list_.back().squaredNorm() + 1e-9f);
            r *= gamma;
        }

        for (int i = 0; i < k; ++i) {
            float beta = rho_list_[i] * y_list_[i].dot(r);
            r += s_list_[i] * (alpha[i] - beta);
        }

        // Store θ_k and ∇f(θ_k) before moving.
        Eigen::VectorXf prev_params = params;
        prev_grad_                  = grad;

        params += lr_ * (-r);

        // s_k is available now; y_k = ∇f(θ_{k+1}) - ∇f(θ_k) will be
        // computed on the next call once the caller supplies the new gradient.
        pending_s_ = params - prev_params;
        has_pending_ = true;
    }

    // Call this at the start of each Step() with the freshly-evaluated grad
    // to commit the previous curvature pair.
    void CommitPair(const Eigen::VectorXf& new_grad) {
        if (!has_pending_) return;
        Eigen::VectorXf y  = new_grad - prev_grad_;
        float           sy = pending_s_.dot(y);
        if (sy > 1e-10f) {
            if (s_list_.size() >= static_cast<std::size_t>(m_)) {
                s_list_.erase(s_list_.begin());
                y_list_.erase(y_list_.begin());
                rho_list_.erase(rho_list_.begin());
            }
            s_list_.push_back(pending_s_);
            y_list_.push_back(y);
            rho_list_.push_back(1.0f / sy);
        }
        has_pending_ = false;
    }

    void Reset() {
        s_list_.clear(); y_list_.clear(); rho_list_.clear();
        has_pending_ = false;
    }

    void SetLr(float lr) { lr_ = lr; }

private:
    int                          m_;
    float                        lr_;
    bool                         has_pending_ = false;
    Eigen::VectorXf              pending_s_;
    Eigen::VectorXf              prev_grad_;
    std::vector<Eigen::VectorXf> s_list_, y_list_;
    std::vector<float>           rho_list_;
};

class ConstrainedOptimizer {
public:
    ConstrainedOptimizer(float inner_lr = 1e-3f, float rho = 1.0f)
        : adam_(inner_lr), rho_(rho) {}

    // Augmented Lagrangian primal step for inequality constraints g_c(θ) ≤ 0.
    //
    // The augmented Lagrangian is:
    //   L_aug = f(θ) + Σ_c [ λ_c·g_c + (ρ/2)·max(0, g_c)² ]
    //
    // Without explicit constraint Jacobians ∇g_c, we scale the task gradient
    // by the total penalty factor:
    //   ∇_θ L_aug ≈ (1 + Σ_c μ_c) · ∇_θ f
    // where μ_c = max(0, λ_c + ρ·g_c) is the active penalty weight.
    //
    // Dual update (projected for inequality: λ ≥ 0):
    //   λ_c ← max(0, λ_c + ρ·g_c)
    void Step(Eigen::VectorXf& params,
              const Eigen::VectorXf& task_grad,
              const std::vector<float>& constraint_violations) {
        if (lambdas_.size() < constraint_violations.size())
            lambdas_.resize(constraint_violations.size(), 0.0f);

        float penalty_scale = 1.0f;
        for (std::size_t c = 0; c < constraint_violations.size(); ++c) {
            float g_c = constraint_violations[c];
            float mu  = std::max(0.0f, lambdas_[c] + rho_ * g_c);
            penalty_scale += mu;
            lambdas_[c] = mu;   // dual update with projection λ ≥ 0
        }

        adam_.Step(params, task_grad * penalty_scale);
    }

    void Reset() { lambdas_.clear(); adam_.Reset(); }

private:
    AdamOptimizer      adam_;
    float              rho_;
    std::vector<float> lambdas_;
};

} // namespace vbx
