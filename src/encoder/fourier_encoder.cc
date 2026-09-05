#include "vbatten_x/encoder.h"
#include "src/encoder/encoder_param.h"
#include "src/field/field_state.h"
#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <random>

namespace vbx {

class FourierEncoder : public PhysicsEncoder {
public:
    FourierEncoder(std::size_t input_dim, std::size_t num_freqs, uint64_t seed)
        : input_dim_(input_dim), num_freqs_(num_freqs) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        freqs_.resize(input_dim_ * num_freqs_);
        for (auto& f : freqs_) f = nd(rng);
        std::size_t out = 2 * num_freqs_;
        std::uniform_real_distribution<float> ud(-0.1f, 0.1f);
        weights_.resize(out);
        for (auto& w : weights_) w = ud(rng);
    }

    FieldState Encode(const PhysicalDataset& ds) override {
        vbx_index   nrows = ds.NumRows();
        std::size_t out   = 2 * num_freqs_;

        std::vector<vbx_float> latent(static_cast<std::size_t>(nrows) * out);
        for (vbx_index r = 0; r < nrows; ++r) {
            auto row = ds.GetRow(r);
            for (std::size_t k = 0; k < num_freqs_; ++k) {
                float dot = 0.0f;
                for (std::size_t j = 0; j < input_dim_ && j < row.size(); ++j)
                    dot += freqs_[j * num_freqs_ + k] * row[j];
                std::size_t base = static_cast<std::size_t>(r) * out + 2 * k;
                latent[base]     = std::sin(dot);
                latent[base + 1] = std::cos(dot);
            }
        }

        FieldState s;
        s.F = MakeContinuousField(out, 1);
        s.F->Embed({latent.data(), out});
        s.K = MakeRegionGraph();
        s.d = MakeUniformDim(static_cast<vbx_dim_t>(out), 1, 1024.0f);
        s.T = MakeRank2Tensor(0, static_cast<vbx_dim_t>(out), static_cast<vbx_dim_t>(out));
        s.bias.assign(static_cast<std::size_t>(nrows), 0.0f);
        return s;
    }

    void UpdateParams(const FieldState&, vbx_float) override {}

private:
    std::size_t        input_dim_;
    std::size_t        num_freqs_;
    std::vector<float> freqs_;
    std::vector<float> weights_;
};

std::unique_ptr<PhysicsEncoder> MakeFourierEncoder(
    std::size_t input_dim, std::size_t num_freqs, uint64_t seed) {
    return std::make_unique<FourierEncoder>(input_dim, num_freqs, seed);
}

} // namespace vbx
