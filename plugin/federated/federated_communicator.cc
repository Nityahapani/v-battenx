#include "vbatten_x/collective/communicator.h"
#include <cmath>
#include <random>
#include <vector>

namespace vbx {

class FederatedCommunicator : public Communicator {
public:
    FederatedCommunicator(int rank, int world_size,
                           float noise_sigma = 0.0f)
        : rank_(rank), world_size_(world_size),
          noise_sigma_(noise_sigma), rng_(rank * 1234567ULL) {}

    int  Rank()      const override { return rank_; }
    int  WorldSize() const override { return world_size_; }

    void Allreduce(std::vector<vbx_float>& data, ReduceOp op) override {
        // signSGD: send sign(gradient) only — differential privacy sketch
        for (auto& v : data) {
            v = (v > 0.0f) ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
            if (noise_sigma_ > 0.0f) {
                std::normal_distribution<float> nd(0.0f, noise_sigma_);
                v += nd(rng_);
            }
        }
    }

    void Broadcast(std::vector<vbx_float>& data, int root) override {
        (void)data; (void)root;
    }

    void Barrier() override {}

    std::string Name() const override { return "federated_signgd"; }

private:
    int              rank_;
    int              world_size_;
    float            noise_sigma_;
    mutable std::mt19937 rng_;
};

std::unique_ptr<Communicator> MakeFederatedCommunicator(int rank,
                                                          int world_size,
                                                          float noise_sigma) {
    return std::make_unique<FederatedCommunicator>(rank, world_size, noise_sigma);
}

} // namespace vbx
