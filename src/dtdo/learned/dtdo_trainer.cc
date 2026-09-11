#include "src/dtdo/learned/dtdo_net_apply.cc"
#include <deque>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace vbx {

struct Experience {
    std::vector<float> region_features;
    std::vector<float> pre_activation;   // GELU input (W1·f + b1), needed for backprop
    std::vector<float> hidden;           // GELU output h, needed for ∂W2
    int                action_idx;
    float              reward;
    float              log_prob;
};

class ExperienceBuffer {
public:
    explicit ExperienceBuffer(std::size_t max_size = 1000)
        : max_size_(max_size) {}

    void Push(Experience e) {
        if (buf_.size() >= max_size_) buf_.pop_front();
        buf_.push_back(std::move(e));
    }

    std::vector<Experience> Sample(std::size_t n, std::mt19937& rng) const {
        std::vector<std::size_t> idx(buf_.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), rng);
        n = std::min(n, buf_.size());
        std::vector<Experience> out;
        out.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
            out.push_back(buf_[idx[i]]);
        return out;
    }

    std::size_t Size() const { return buf_.size(); }

private:
    std::deque<Experience> buf_;
    std::size_t            max_size_;
};

// GELU derivative: GELU'(x) = 0.5*(1 + tanh(c*(x+0.044715*x³)))
//                             + 0.5*x * sech²(c*(x+0.044715*x³)) * c*(1+3*0.044715*x²)
// where c = 0.7978845608.
static Eigen::VectorXf GeluGrad(const Eigen::VectorXf& pre) {
    const float c = 0.7978845608f;
    Eigen::VectorXf out(pre.size());
    for (int i = 0; i < pre.size(); ++i) {
        float x  = pre[i];
        float u  = c * (x + 0.044715f * x * x * x);
        float t  = std::tanh(u);
        float du = c * (1.0f + 3.0f * 0.044715f * x * x);
        out[i]   = 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
    }
    return out;
}

class DtdoTrainer {
public:
    DtdoTrainer(DtdoNet*  net,
                float     lr          = 1e-3f,
                float     gamma       = 0.99f,
                int       update_freq = 10,
                std::size_t buf_size  = 1000)
        : net_(net), lr_(lr), gamma_(gamma),
          update_freq_(update_freq), buffer_(buf_size), rng_(42) {}

    // Record a transition. Caller must supply pre_activation (W1·f+b1) and
    // hidden (GELU output) from the forward pass so Update() can backprop.
    void RecordTransition(const std::vector<float>& feat,
                           const std::vector<float>& pre_activation,
                           const std::vector<float>& hidden,
                           int    action_idx,
                           float  reward,
                           float  log_prob) {
        buffer_.Push({feat, pre_activation, hidden, action_idx, reward, log_prob});
        ++step_count_;
    }

    void MaybeUpdate() {
        if (step_count_ % update_freq_ == 0 && buffer_.Size() >= 32)
            Update();
    }

    // REINFORCE update with baseline (mean reward).
    //
    // For each experience (f, pre, h, a, r):
    //   Â = (r - mean_r) / std_r                       [normalised advantage]
    //
    // Two-layer head:  pre = W1·f + b1,  h = GELU(pre),  logit = W2·h + b2
    //
    // Policy-gradient loss:  L = -E[log π(a|s) · Â]
    //
    // Key identity: d(log π(a))/d(logits) = e_a - π
    // Therefore:   dL/d(logits) = -Â·(e_a - π) = Â·(π - e_a)
    //
    // Gradient (per sample, averaged over batch):
    //   δ_logit = Â·(π - e_a) / B
    //   ∂L/∂W2 += δ_logit · hᵀ
    //   ∂L/∂b2 += δ_logit
    //   δ_h    = W2ᵀ · δ_logit
    //   δ_pre  = δ_h ⊙ GELU'(pre)
    //   ∂L/∂W1 += δ_pre · fᵀ
    //   ∂L/∂b1 += δ_pre
    void Update() {
        auto batch = buffer_.Sample(32, rng_);
        float bs   = static_cast<float>(batch.size());

        float mean_r = 0.0f;
        for (auto& e : batch) mean_r += e.reward;
        mean_r /= bs;

        float std_r = 0.0f;
        for (auto& e : batch) std_r += (e.reward - mean_r) * (e.reward - mean_r);
        std_r = std::sqrt(std_r / bs + 1e-8f);

        auto& head   = net_->HeadLayer();
        auto& logits = net_->LogitLayer();

        int hd   = static_cast<int>(head.W.rows());
        int fd   = static_cast<int>(head.W.cols());
        int od   = static_cast<int>(logits.W.rows());

        Eigen::MatrixXf dW1 = Eigen::MatrixXf::Zero(hd, fd);
        Eigen::VectorXf db1 = Eigen::VectorXf::Zero(hd);
        Eigen::MatrixXf dW2 = Eigen::MatrixXf::Zero(od, hd);
        Eigen::VectorXf db2 = Eigen::VectorXf::Zero(od);

        float pg_loss = 0.0f;

        for (auto& e : batch) {
            float adv = (e.reward - mean_r) / std_r;

            Eigen::Map<const Eigen::VectorXf> f(
                e.region_features.data(), static_cast<int>(e.region_features.size()));
            Eigen::Map<const Eigen::VectorXf> pre(
                e.pre_activation.data(), static_cast<int>(e.pre_activation.size()));
            Eigen::Map<const Eigen::VectorXf> h(
                e.hidden.data(), static_cast<int>(e.hidden.size()));

            // Re-compute logits and softmax (W2 may have changed since recording).
            Eigen::VectorXf logit_vec = logits.Forward(h);
            int n_act = static_cast<int>(logit_vec.size());
            Eigen::VectorXf pi = (logit_vec.array() - logit_vec.maxCoeff()).exp();
            pi /= pi.sum();

            pg_loss -= (e.action_idx < n_act ? std::log(pi[e.action_idx] + 1e-9f) : 0.0f) * adv;

            // dL/dlogits = Â · (π − e_a) / B
            // d(log π(a))/dlogits = e_a − π  ⟹  dL/dlogits = −Â·(e_a−π) = Â·(π−e_a)
            Eigen::VectorXf delta_logit = pi;
            if (e.action_idx < n_act) delta_logit[e.action_idx] -= 1.0f;
            delta_logit *= adv / bs;

            dW2 += delta_logit * h.transpose();
            db2 += delta_logit;

            Eigen::VectorXf delta_h   = logits.W.transpose() * delta_logit;
            Eigen::VectorXf delta_pre = delta_h.cwiseProduct(GeluGrad(pre));

            if (delta_pre.size() == hd && f.size() == fd) {
                dW1 += delta_pre * f.transpose();
                db1 += delta_pre;
            }
        }

        head.W   -= lr_ * dW1;
        head.b   -= lr_ * db1;
        logits.W -= lr_ * dW2;
        logits.b -= lr_ * db2;

        total_updates_++;
        if (total_updates_ % 50 == 0)
            std::cout << "[DTDO-trainer] update=" << total_updates_
                      << " pg_loss=" << pg_loss / bs << "\n";
    }

    void ImmitationStep(const std::vector<float>& feat,
                         int rule_based_action) {
        Eigen::Map<const Eigen::VectorXf> f(feat.data(),
                                              static_cast<int>(feat.size()));
        auto& head   = net_->HeadLayer();
        auto& logits = net_->LogitLayer();

        Eigen::VectorXf pre = head.Forward(f);
        Eigen::VectorXf h   = Gelu(pre);
        Eigen::VectorXf out = logits.Forward(h);

        if (rule_based_action >= out.size()) return;

        Eigen::VectorXf probs = (out.array() - out.maxCoeff()).exp();
        probs /= probs.sum();

        // Cross-entropy gradient w.r.t. logits: π - e_a
        Eigen::VectorXf delta_logit = probs;
        delta_logit[rule_based_action] -= 1.0f;
        delta_logit *= lr_;

        logits.W -= delta_logit * h.transpose();
        logits.b -= delta_logit;

        Eigen::VectorXf delta_h   = logits.W.transpose() * delta_logit;
        Eigen::VectorXf delta_pre = delta_h.cwiseProduct(GeluGrad(pre));
        head.W -= delta_pre * f.transpose();
        head.b -= delta_pre;
    }

    int TotalUpdates() const { return total_updates_; }

private:
    DtdoNet*          net_;
    float             lr_;
    float             gamma_;
    int               update_freq_;
    ExperienceBuffer  buffer_;
    std::mt19937      rng_;
    int               step_count_   = 0;
    int               total_updates_ = 0;
};

} // namespace vbx
