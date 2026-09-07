#include "vbatten_x/topological_operator.h"
#include "vbatten_x/mutation_result.h"
#include "src/field/field_state.h"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <limits>
#include <random>
#include <algorithm>

namespace vbx {

struct Action {
    MutationType  type;
    vbx_region_id region;
    vbx_dim_t     target_dim;
    int           split_axis;
    vbx_region_id merge_target;
};

class ActionSpace {
public:
    explicit ActionSpace(int num_regions, int num_mutation_types = 10)
        : num_regions_(num_regions), num_types_(num_mutation_types) {}

    std::vector<Action> AllActions() const {
        std::vector<Action> actions;
        actions.push_back({MutationType::NoOp, 0, 0, 0, 0});
        for (int r = 0; r < num_regions_; ++r) {
            actions.push_back({MutationType::Expand1dTo2d,   (vbx_region_id)r, 2, 0, 0});
            actions.push_back({MutationType::Expand2dTo3d,   (vbx_region_id)r, 3, 0, 0});
            actions.push_back({MutationType::Collapse3dTo2d, (vbx_region_id)r, 2, 0, 0});
            actions.push_back({MutationType::Collapse2dTo1d, (vbx_region_id)r, 1, 0, 0});
            actions.push_back({MutationType::LocalDimChange,  (vbx_region_id)r, 2, 0, 0});
            actions.push_back({MutationType::SplitRegion,    (vbx_region_id)r, 0, 0, 0});
        }
        return actions;
    }

    int NumActions() const { return 1 + num_regions_ * 6; }

    void MaskInvalidActions(Eigen::VectorXf& logits,
                             const ComplexityCost& budget) const {
        auto actions = AllActions();
        for (int i = 0; i < static_cast<int>(actions.size()) && i < logits.size(); ++i)
            if (!budget.CanAfford(actions[i].type))
                logits[i] = -std::numeric_limits<float>::infinity();
    }

    Action Greedy(const Eigen::VectorXf& logits,
                  const ComplexityCost& budget) const {
        auto actions = AllActions();
        Eigen::VectorXf masked = logits;
        MaskInvalidActions(masked, budget);
        int best = 0;
        float best_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < masked.size(); ++i)
            if (masked[i] > best_val) { best_val = masked[i]; best = i; }
        if (best < static_cast<int>(actions.size())) return actions[best];
        return {MutationType::NoOp, 0, 0, 0, 0};
    }

    Action Sample(const Eigen::VectorXf& logits,
                  const ComplexityCost& budget,
                  float temperature,
                  std::mt19937& rng) const {
        auto actions = AllActions();
        Eigen::VectorXf masked = logits;
        MaskInvalidActions(masked, budget);

        Eigen::VectorXf probs = (masked / temperature).array().exp();
        float sum = probs.sum();
        if (sum < 1e-9f) return {MutationType::NoOp, 0, 0, 0, 0};
        probs /= sum;

        std::uniform_real_distribution<float> ud(0.0f, 1.0f);
        float r = ud(rng);
        float cum = 0.0f;
        for (int i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (r <= cum && i < static_cast<int>(actions.size()))
                return actions[i];
        }
        return actions.back();
    }

private:
    int num_regions_;
    int num_types_;
};

} // namespace vbx
