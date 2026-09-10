#include <random>
#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace vbx {

static Eigen::MatrixXf PcaBasis(const Eigen::MatrixXf& data, int target_dim) {
    Eigen::MatrixXf centered = data.rowwise() - data.colwise().mean();
    Eigen::MatrixXf cov      = centered.transpose() * centered / std::max(1, (int)data.rows() - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig(cov);
    int k = std::min(target_dim, (int)eig.eigenvectors().cols());
    return eig.eigenvectors().rightCols(k);
}

static void NormaliseFrobenius(std::vector<float>& v, float original_norm) {
    double n = 0.0;
    for (float x : v) n += x * x;
    n = std::sqrt(n);
    if (n > 1e-12 && original_norm > 1e-12) {
        float scale = original_norm / static_cast<float>(n);
        for (auto& x : v) x *= scale;
    }
}

std::vector<float> Expand1dTo2d(const std::vector<float>& params, uint64_t seed) {
    float norm = 0.0f;
    for (float v : params) norm += v * v;
    norm = std::sqrt(norm);

    std::vector<float> out(params.size() * 2);
    for (std::size_t i = 0; i < params.size(); ++i) {
        out[2 * i]     = params[i];
        out[2 * i + 1] = 0.0f;
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.01f);
    for (std::size_t i = 1; i < out.size(); i += 2)
        out[i] = nd(rng);

    NormaliseFrobenius(out, norm);
    return out;
}

std::vector<float> Expand2dTo3d(const std::vector<float>& params, uint64_t seed) {
    float norm = 0.0f;
    for (float v : params) norm += v * v;
    norm = std::sqrt(norm);

    std::size_t n2    = params.size();
    std::size_t pairs = n2 / 2;
    std::vector<float> out(pairs * 3, 0.0f);
    for (std::size_t i = 0; i < pairs; ++i) {
        out[3 * i]     = params[2 * i];
        out[3 * i + 1] = params[2 * i + 1];
        out[3 * i + 2] = 0.0f;
    }

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 0.01f);
    for (std::size_t i = 2; i < out.size(); i += 3)
        out[i] = nd(rng);

    NormaliseFrobenius(out, norm);
    return out;
}

std::vector<float> Collapse3dTo2d(const std::vector<float>& params) {
    std::size_t triples = params.size() / 3;
    Eigen::MatrixXf data(triples, 3);
    for (std::size_t i = 0; i < triples; ++i)
        for (int j = 0; j < 3; ++j)
            data(i, j) = params[3 * i + j];

    auto basis           = PcaBasis(data, 2);
    Eigen::MatrixXf proj = data * basis;

    // PCA projection already minimises reconstruction error (Eckart–Young).
    // Do not rescale: proj's Frobenius norm equals sqrt(sum of top-2
    // eigenvalues), which is the energy that the 2-d subspace can represent.
    std::vector<float> out(triples * 2);
    for (std::size_t i = 0; i < triples; ++i)
        for (int j = 0; j < 2; ++j)
            out[2 * i + j] = proj(i, j);
    return out;
}

std::vector<float> Collapse2dTo1d(const std::vector<float>& params) {
    std::size_t pairs = params.size() / 2;
    Eigen::MatrixXf data(pairs, 2);
    for (std::size_t i = 0; i < pairs; ++i)
        for (int j = 0; j < 2; ++j)
            data(i, j) = params[2 * i + j];

    auto basis           = PcaBasis(data, 1);
    Eigen::MatrixXf proj = data * basis;

    std::vector<float> out(pairs);
    for (std::size_t i = 0; i < pairs; ++i)
        out[i] = proj(i, 0);
    return out;
}

std::vector<float> LocalDimChange(const std::vector<float>& params,
                                   vbx_dim_t from_d, vbx_dim_t to_d,
                                   uint64_t seed) {
    if (to_d == from_d) return params;
    if (to_d > from_d) {
        if (from_d == 1 && to_d == 2) return Expand1dTo2d(params, seed);
        if (from_d == 2 && to_d == 3) return Expand2dTo3d(params, seed);
        float norm = 0.0f;
        for (float v : params) norm += v * v;
        norm = std::sqrt(norm);
        std::size_t n = params.size();
        std::size_t groups = n / from_d;
        std::vector<float> out(groups * to_d, 0.0f);
        for (std::size_t g = 0; g < groups; ++g)
            for (int d = 0; d < from_d; ++d)
                out[g * to_d + d] = params[g * from_d + d];
        NormaliseFrobenius(out, norm);
        return out;
    } else {
        if (from_d == 3 && to_d == 2) return Collapse3dTo2d(params);
        if (from_d == 2 && to_d == 1) return Collapse2dTo1d(params);
        std::size_t n      = params.size();
        std::size_t groups = n / from_d;
        Eigen::MatrixXf data(groups, from_d);
        for (std::size_t g = 0; g < groups; ++g)
            for (int d = 0; d < from_d; ++d)
                data(g, d) = params[g * from_d + d];
        auto basis           = PcaBasis(data, to_d);
        Eigen::MatrixXf proj = data * basis;
        std::vector<float> out(groups * to_d);
        for (std::size_t g = 0; g < groups; ++g)
            for (int d = 0; d < to_d; ++d)
                out[g * to_d + d] = proj(g, d);
        return out;
    }
}

} // namespace vbx
