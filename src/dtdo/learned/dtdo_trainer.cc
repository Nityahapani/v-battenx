#include "src/dtdo/learned/dtdo_net_apply.cc"
#include <deque>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace vbx {

struct Experience {
    std::vector<float> region_features;
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

class DtdoTrainer {
public:
    DtdoTrainer(DtdoNet*  net,
                float     lr          = 1e-3f,
                float     gamma       = 0.99f,
                int       update_freq = 10,
                std::size_t buf_size  = 1000)
        : net_(net), lr_(lr), gamma_(gamma),
          update_freq_(update_freq), buffer_(buf_size), rng_(42) {}

    void RecordTransition(const std::vector<float>& feat,
                           int    action_idx,
                           float  reward,
                           float  log_prob) {
        buffer_.Push({feat, action_idx, reward, log_prob});
        ++step_count_;
    }

    void MaybeUpdate() {
        if (step_count_ % update_freq_ == 0 && buffer_.Size() >= 32)
            Update();
    }

    void Update() {
        auto batch    = buffer_.Sample(32, rng_);
        float pg_loss = 0.0f;

        float mean_r = 0.0f;
        for (auto& e : batch) mean_r += e.reward;
        mean_r /= batch.size();

        float std_r = 0.0f;
        for (auto& e : batch) std_r += (e.reward - mean_r) * (e.reward - mean_r);
        std_r = std::sqrt(std_r / batch.size() + 1e-8f);

        for (auto& e : batch) {
            float adv = (e.reward - mean_r) / std_r;
            pg_loss  -= e.log_prob * adv;
        }
        pg_loss /= batch.size();

        // gradient step on head + logit layers
        auto& head   = net_->HeadLayer();
        auto& logits = net_->LogitLayer();

        float grad_scale = -lr_ * pg_loss;
        head.W   *= (1.0f - grad_scale * 1e-4f);
        logits.W *= (1.0f - grad_scale * 1e-4f);

        total_updates_++;
        if (total_updates_ % 50 == 0)
            std::cout << "[DTDO-trainer] update=" << total_updates_
                      << " pg_loss=" << pg_loss << "\n";
    }

    void ImmitationStep(const std::vector<float>& feat,
                         int rule_based_action) {
        Eigen::Map<const Eigen::VectorXf> f(feat.data(),
                                              static_cast<int>(feat.size()));
        auto& head   = net_->HeadLayer();
        auto& logits = net_->LogitLayer();

        Eigen::VectorXf h   = Gelu(head.Forward(f));
        Eigen::VectorXf out = logits.Forward(h);

        if (rule_based_action >= out.size()) return;

        Eigen::VectorXf probs = (out.array() - out.maxCoeff()).exp();
        probs /= probs.sum();

        float log_p = std::log(probs[rule_based_action] + 1e-9f);
        float ce    = -log_p;

        // supervised step: pull logit for correct action up
        Eigen::VectorXf grad_out = probs;
        grad_out[rule_based_action] -= 1.0f;
        grad_out *= lr_;

        logits.W -= grad_out * h.transpose();
        logits.b -= grad_out;
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
