#include "vbatten_x/encoder.h"
#include "src/encoder/encoder_param.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

namespace vbx {

static Eigen::VectorXf ApplyAct(const Eigen::VectorXf& x, Activation act) {
    if (act == Activation::ReLU)  return x.cwiseMax(0.0f);
    if (act == Activation::Tanh)  return x.array().tanh();
    // GELU
    Eigen::VectorXf out(x.size());
    for (int i = 0; i < x.size(); ++i) {
        float v = x[i];
        out[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    return out;
}

struct Layer {
    Eigen::MatrixXf W;
    Eigen::VectorXf b;

    Layer(std::size_t in, std::size_t out, std::mt19937& rng) {
        float scale = std::sqrt(2.0f / static_cast<float>(in));
        std::normal_distribution<float> nd(0.0f, scale);
        W = Eigen::MatrixXf(out, in).unaryExpr([&](float) { return nd(rng); });
        b = Eigen::VectorXf::Zero(static_cast<int>(out));
    }

    Eigen::VectorXf Forward(const Eigen::VectorXf& x) const { return W * x + b; }
};

class MlpEncoder : public PhysicsEncoder {
public:
    MlpEncoder(std::size_t input_dim, EncoderParam p) : param_(p) {
        std::mt19937 rng(p.rng_seed);
        std::size_t prev = input_dim;
        for (int l = 0; l < p.num_layers - 1; ++l) {
            layers_.emplace_back(prev, p.hidden_dim, rng);
            prev = p.hidden_dim;
        }
        layers_.emplace_back(prev, p.latent_dim, rng);
    }

    FieldState Encode(const PhysicalDataset& ds) override {
        vbx_index nrows = ds.NumRows();
        std::size_t ldim = param_.latent_dim;

        std::vector<vbx_float> latent(static_cast<std::size_t>(nrows) * ldim);
        for (vbx_index r = 0; r < nrows; ++r) {
            auto row = ds.GetRow(r);
            Eigen::Map<const Eigen::VectorXf> x(row.data(), static_cast<int>(row.size()));
            Eigen::VectorXf h = x;
            for (std::size_t l = 0; l < layers_.size() - 1; ++l)
                h = ApplyAct(layers_[l].Forward(h), param_.activation);
            h = layers_.back().Forward(h);
            for (std::size_t j = 0; j < ldim; ++j)
                latent[static_cast<std::size_t>(r) * ldim + j] = h[static_cast<int>(j)];
        }

        FieldState s;
        s.F = MakeContinuousField(ldim, 1);
        s.F->Embed({latent.data(), ldim});
        s.K = MakeRegionGraph();
        s.d = MakeUniformDim(static_cast<vbx_dim_t>(ldim), 1, 1024.0f);
        s.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(ldim), static_cast<vbx_dim_t>(ldim));
        s.bias.assign(static_cast<std::size_t>(nrows), 0.0f);
        return s;
    }

    void UpdateParams(const FieldState&, vbx_float) override {}

private:
    EncoderParam        param_;
    std::vector<Layer>  layers_;
};

std::unique_ptr<PhysicsEncoder> MakeMlpEncoder(std::size_t input_dim, EncoderParam p) {
    return std::make_unique<MlpEncoder>(input_dim, p);
}

} // namespace vbx
