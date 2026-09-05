#include "vbatten_x/encoder.h"
#include "encoder_param.h"
#include "src/field/field_state.h"
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <vector>

namespace vbx {

static Eigen::VectorXf ApplyActivation(const Eigen::VectorXf& x, Activation act) {
    switch (act) {
        case Activation::ReLU:
            return x.cwiseMax(0.0f);
        case Activation::GELU: {
            Eigen::VectorXf out(x.size());
            for (int i = 0; i < x.size(); ++i) {
                float v = x[i];
                out[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
            }
            return out;
        }
        case Activation::Tanh:
            return x.array().tanh();
    }
    return x;
}

struct LinearLayer {
    Eigen::MatrixXf W;
    Eigen::VectorXf b;

    LinearLayer(std::size_t in, std::size_t out, std::mt19937& rng) {
        float scale = std::sqrt(2.0f / in);
        std::normal_distribution<float> dist(0.0f, scale);
        W = Eigen::MatrixXf(out, in).unaryExpr([&](float) { return dist(rng); });
        b = Eigen::VectorXf::Zero(out);
    }

    Eigen::VectorXf Forward(const Eigen::VectorXf& x) const { return W * x + b; }
};

class MlpEncoder : public PhysicsEncoder {
public:
    explicit MlpEncoder(std::size_t input_dim, EncoderParam p) : param_(p) {
        std::mt19937 rng(p.rng_seed);
        std::size_t prev = input_dim;
        for (int l = 0; l < p.num_layers - 1; ++l) {
            layers_.emplace_back(prev, p.hidden_dim, rng);
            prev = p.hidden_dim;
        }
        layers_.emplace_back(prev, p.latent_dim, rng);
    }

    FieldState Encode(const PhysicalDataset& ds) override;
    void UpdateParams(const FieldState&, vbx_float) override {}

private:
    Eigen::VectorXf Forward(Span<const vbx_float> row) const {
        Eigen::Map<const Eigen::VectorXf> x(row.data(), row.size());
        Eigen::VectorXf h = x;
        for (std::size_t l = 0; l < layers_.size() - 1; ++l)
            h = ApplyActivation(layers_[l].Forward(h), param_.activation);
        return layers_.back().Forward(h);
    }

    EncoderParam            param_;
    std::vector<LinearLayer> layers_;
};

} // namespace vbx

#include "src/field/manifold/continuous_field.cc"
#include "src/field/topology/region_graph.cc"
#include "src/field/dimension/uniform_dim.cc"
#include "src/field/tensor/contraction_engine.cc"

namespace vbx {

extern std::unique_ptr<LatentField>   MakeContinuousField(std::size_t, std::size_t);
extern std::unique_ptr<FieldTopology> MakeRegionGraph();
extern std::unique_ptr<DimensionMap>  MakeUniformDim(vbx_dim_t, std::size_t, float);
extern std::unique_ptr<TensorField>   MakeRank2Tensor(vbx_region_id, vbx_dim_t, vbx_dim_t);

FieldState MlpEncoder::Encode(const PhysicalDataset& ds) {
    vbx_index nrows = ds.NumRows();
    std::size_t ldim = param_.latent_dim;

    std::vector<vbx_float> latent(nrows * ldim);
    for (vbx_index r = 0; r < nrows; ++r) {
        Eigen::VectorXf h = Forward(ds.GetRow(r));
        for (std::size_t j = 0; j < ldim; ++j)
            latent[r * ldim + j] = h[j];
    }

    FieldState s;
    s.F = MakeContinuousField(ldim, 1);
    s.F->Embed({latent.data(), ldim});
    s.K = MakeRegionGraph();
    s.d = MakeUniformDim(static_cast<vbx_dim_t>(ldim), 1, 1024.0f);
    s.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(ldim), static_cast<vbx_dim_t>(ldim));
    s.bias.assign(nrows, 0.0f);
    return s;
}

std::unique_ptr<PhysicsEncoder> MakeMlpEncoder(std::size_t input_dim, EncoderParam p) {
    return std::make_unique<MlpEncoder>(input_dim, p);
}

} // namespace vbx
