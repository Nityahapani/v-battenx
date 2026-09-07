#include "vbatten_x/topological_operator.h"
#include "vbatten_x/physics_evaluator.h"
#include "src/field/field_state.h"
#include "src/field/topology/simplex_complex.h"
#include "src/dtdo/learned/mutation_policy.cc"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

namespace vbx {

struct LayerNorm {
    Eigen::VectorXf gamma;
    Eigen::VectorXf beta;
    float eps = 1e-5f;

    LayerNorm(int dim) : gamma(Eigen::VectorXf::Ones(dim)),
                          beta(Eigen::VectorXf::Zero(dim)) {}

    Eigen::VectorXf Forward(const Eigen::VectorXf& x) const {
        float mean = x.mean();
        float var  = (x.array() - mean).square().mean();
        return ((x.array() - mean) / std::sqrt(var + eps)).matrix()
                   .cwiseProduct(gamma) + beta;
    }
};

static Eigen::VectorXf Gelu(const Eigen::VectorXf& x) {
    Eigen::VectorXf out(x.size());
    for (int i = 0; i < x.size(); ++i) {
        float v = x[i];
        out[i]  = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    return out;
}

struct LinearLayer {
    Eigen::MatrixXf W;
    Eigen::VectorXf b;

    LinearLayer(int in, int out, std::mt19937& rng) {
        float s = std::sqrt(2.0f / in);
        std::normal_distribution<float> nd(0.0f, s);
        W = Eigen::MatrixXf(out, in).unaryExpr([&](float) { return nd(rng); });
        b = Eigen::VectorXf::Zero(out);
    }

    Eigen::VectorXf Forward(const Eigen::VectorXf& x) const { return W * x + b; }
};

struct RegionMLP {
    LinearLayer l1;
    LinearLayer l2;
    LayerNorm   ln1;
    LayerNorm   ln2;

    RegionMLP(int in, int hidden, std::mt19937& rng)
        : l1(in, hidden, rng), l2(hidden, hidden, rng),
          ln1(hidden), ln2(hidden) {}

    Eigen::VectorXf Forward(const Eigen::VectorXf& x) const {
        return ln2.Forward(Gelu(l2.Forward(ln1.Forward(Gelu(l1.Forward(x))))));
    }
};

static Eigen::VectorXf BuildRegionFeatures(const FieldState&    state,
                                             const ResidualInfo&  residuals,
                                             vbx_region_id        rid,
                                             const ComplexityCost& budget) {
    float pde_r    = residuals.pde_residual.empty()  ? 0.0f : residuals.pde_residual[0];
    float pred_r   = residuals.prediction_residual.empty() ? 0.0f : residuals.prediction_residual[0];
    float local_d  = state.d ? static_cast<float>(state.d->LocalDim(rid)) : 1.0f;
    float bud_rem  = static_cast<float>(budget.BudgetRemaining());
    float num_nb   = state.K ? static_cast<float>(state.K->Neighbours(rid).size()) : 0.0f;

    SimplexComplex sc{*state.K};
    float b0 = static_cast<float>(sc.Betti0());
    float b1 = static_cast<float>(sc.Betti1());

    auto params = state.F ? state.F->Params() : std::vector<vbx_float>{};
    float mean_embed = 0.0f;
    if (!params.empty()) {
        for (float v : params) mean_embed += v;
        mean_embed /= params.size();
    }

    Eigen::VectorXf feat(8);
    feat << pde_r, pred_r, local_d, bud_rem, num_nb, b0, b1, mean_embed;
    return feat;
}

class DtdoNet : public TopologicalOperator {
public:
    DtdoNet(int hidden_dim = 64, PolicyType policy = PolicyType::Stochastic,
            float temperature = 1.0f, uint64_t seed = 42)
        : hidden_(hidden_dim), policy_(policy, temperature), rng_(seed)
    {
        std::mt19937 init_rng(seed);
        region_mlp_ = std::make_unique<RegionMLP>(8, hidden_dim, init_rng);
        head_        = std::make_unique<LinearLayer>(hidden_dim * 2, 64, init_rng);
        logit_layer_ = std::make_unique<LinearLayer>(64, 64, init_rng);
    }

    MutationResult Apply(const FieldState&     current,
                          const ResidualInfo&  residuals,
                          const ComplexityCost& budget) const override {
        int num_regions = current.K ? static_cast<int>(current.K->NumRegions()) : 1;
        ActionSpace space(num_regions);

        Eigen::MatrixXf region_embs(hidden_, num_regions);
        for (int r = 0; r < num_regions; ++r) {
            auto feat = BuildRegionFeatures(current, residuals,
                                             static_cast<vbx_region_id>(r), budget);
            region_embs.col(r) = region_mlp_->Forward(feat);
        }

        Eigen::VectorXf global_max  = region_embs.rowwise().maxCoeff();
        Eigen::VectorXf global_mean = region_embs.rowwise().mean();
        Eigen::VectorXf global(hidden_ * 2);
        global << global_max, global_mean;

        Eigen::VectorXf h      = Gelu(head_->Forward(global));
        Eigen::VectorXf logits = logit_layer_->Forward(h);

        if (logits.size() < space.NumActions()) {
            Eigen::VectorXf padded = Eigen::VectorXf::Zero(space.NumActions());
            padded.head(logits.size()) = logits;
            logits = padded;
        } else {
            logits = logits.head(space.NumActions());
        }

        Action act = policy_.Select(logits, budget, space);

        if (act.type == MutationType::NoOp) {
            MutationResult r;
            r.next    = current.Clone();
            r.mutated = false;
            return r;
        }

        return ApplyAction(current, act);
    }

    std::string Name() const override { return "learned_dtdo"; }

    const LinearLayer& HeadLayer()   const { return *head_; }
    LinearLayer&       HeadLayer()         { return *head_; }
    const LinearLayer& LogitLayer()  const { return *logit_layer_; }
    LinearLayer&       LogitLayer()        { return *logit_layer_; }

private:
    MutationResult ApplyAction(const FieldState& s, const Action& act) const;

    int                           hidden_;
    mutable MutationPolicy        policy_;
    mutable std::mt19937          rng_;
    std::unique_ptr<RegionMLP>    region_mlp_;
    std::unique_ptr<LinearLayer>  head_;
    std::unique_ptr<LinearLayer>  logit_layer_;
};

} // namespace vbx
