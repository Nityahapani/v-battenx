#include "src/dtdo/learned/action_space.cc"
#include <random>

namespace vbx {

enum class PolicyType { Greedy, Stochastic };

class MutationPolicy {
public:
    MutationPolicy(PolicyType type = PolicyType::Greedy, float temperature = 1.0f)
        : type_(type), temperature_(temperature), rng_(42) {}

    Action Select(const Eigen::VectorXf& logits,
                  const ComplexityCost&  budget,
                  const ActionSpace&     space) {
        if (type_ == PolicyType::Greedy)
            return space.Greedy(logits, budget);
        return space.Sample(logits, budget, temperature_, rng_);
    }

    void SetTemperature(float t)  { temperature_ = t; }
    void SetSeed(uint64_t seed)   { rng_.seed(seed); }

private:
    PolicyType   type_;
    float        temperature_;
    std::mt19937 rng_;
};

} // namespace vbx
