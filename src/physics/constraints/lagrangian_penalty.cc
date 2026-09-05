#include "src/field/field_state.h"
#include <vector>
#include <cmath>

namespace vbx {

class LagrangianPenalty {
public:
    explicit LagrangianPenalty(float rho = 1.0f) : rho_(rho) {}

    void AddConstraint(float violation) {
        if (lambdas_.size() < violations_.size() + 1)
            lambdas_.push_back(0.0f);
        violations_.push_back(violation);
    }

    float Penalty() const {
        float total = 0.0f;
        for (std::size_t i = 0; i < violations_.size(); ++i) {
            float v = violations_[i];
            float lam = i < lambdas_.size() ? lambdas_[i] : 0.0f;
            total += lam * v + 0.5f * rho_ * v * v;
        }
        return total;
    }

    void DualUpdate() {
        for (std::size_t i = 0; i < violations_.size(); ++i) {
            if (i < lambdas_.size())
                lambdas_[i] += rho_ * violations_[i];
        }
        violations_.clear();
    }

    void Reset() { lambdas_.clear(); violations_.clear(); }

private:
    float              rho_;
    std::vector<float> lambdas_;
    std::vector<float> violations_;
};

} // namespace vbx
