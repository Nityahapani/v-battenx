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

    void Step(Eigen::VectorXf& params,
              const Eigen::VectorXf& grad,
              std::function<float(const Eigen::VectorXf&)> loss_fn) {
        if (s_list_.size() >= static_cast<std::size_t>(m_)) {
            s_list_.erase(s_list_.begin());
            y_list_.erase(y_list_.begin());
            rho_list_.erase(rho_list_.begin());
        }

        Eigen::VectorXf q = grad;
        int k = static_cast<int>(s_list_.size());
        std::vector<float> alpha(k);

        for (int i = k - 1; i >= 0; --i) {
            alpha[i] = rho_list_[i] * s_list_[i].dot(q);
            q -= alpha[i] * y_list_[i];
        }

        Eigen::VectorXf r = q;
        if (k > 0) {
            float gamma = s_list_.back().dot(y_list_.back())
                        / (y_list_.back().squaredNorm() + 1e-9f);
            r *= gamma;
        }

        for (int i = 0; i < k; ++i) {
            float beta = rho_list_[i] * y_list_[i].dot(r);
            r += s_list_[i] * (alpha[i] - beta);
        }

        Eigen::VectorXf direction = -r;
        Eigen::VectorXf prev      = params;
        Eigen::VectorXf prev_grad = grad;

        params += lr_ * direction;

        Eigen::VectorXf s   = params - prev;
        Eigen::VectorXf y   = grad - prev_grad;
        float sy = s.dot(y);
        if (sy > 1e-10f) {
            s_list_.push_back(s);
            y_list_.push_back(y);
            rho_list_.push_back(1.0f / sy);
        }
    }

    void Reset() { s_list_.clear(); y_list_.clear(); rho_list_.clear(); }

private:
    int                      m_;
    float                    lr_;
    std::vector<Eigen::VectorXf> s_list_, y_list_;
    std::vector<float>           rho_list_;
};

class ConstrainedOptimizer {
public:
    ConstrainedOptimizer(float inner_lr = 1e-3f,
                          float rho     = 1.0f,
                          int   inner_steps = 10)
        : adam_(inner_lr), rho_(rho), inner_steps_(inner_steps) {}

    void Step(Eigen::VectorXf& params,
              const Eigen::VectorXf& task_grad,
              const std::vector<float>& constraint_violations) {
        for (std::size_t c = 0; c < constraint_violations.size(); ++c) {
            if (lambdas_.size() <= c) lambdas_.push_back(0.0f);
            lambdas_[c] += rho_ * constraint_violations[c];
        }

        Eigen::VectorXf aug_grad = task_grad;
        for (std::size_t c = 0; c < constraint_violations.size() && c < lambdas_.size(); ++c) {
            float penalty_grad = lambdas_[c] + rho_ * constraint_violations[c];
            aug_grad.array()  += penalty_grad * 0.01f;
        }

        for (int step = 0; step < inner_steps_; ++step)
            adam_.Step(params, aug_grad);
    }

    void Reset() { lambdas_.clear(); adam_.Reset(); }

private:
    AdamOptimizer      adam_;
    float              rho_;
    int                inner_steps_;
    std::vector<float> lambdas_;
};

} // namespace vbx
