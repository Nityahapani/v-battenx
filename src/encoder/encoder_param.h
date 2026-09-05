#pragma once

#include <string>
#include <cstddef>

namespace vbx {

enum class Activation { ReLU, GELU, Tanh };

struct EncoderParam {
    std::size_t hidden_dim  = 64;
    std::size_t latent_dim  = 32;
    int         num_layers  = 3;
    Activation  activation  = Activation::GELU;
    float       dropout     = 0.0f;
    uint64_t    rng_seed    = 42;
};

} // namespace vbx
