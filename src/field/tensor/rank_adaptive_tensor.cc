#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>

namespace vbx {

class RankAdaptiveTensor : public TensorField {
public:
    RankAdaptiveTensor(vbx_region_id region, vbx_dim_t dim)
        : region_(region), dim_(dim),
          mat_(Eigen::MatrixXf::Identity(dim, dim)) {}

    int           Rank()   const override { return 2; }
    std::size_t   Size()   const override { return static_cast<std::size_t>(mat_.size()); }
    vbx_region_id Region() const override { return region_; }

    Span<const vbx_float> Data() const override {
        return {mat_.data(), static_cast<std::size_t>(mat_.size())};
    }
    Span<vbx_float> Data() override {
        return {mat_.data(), static_cast<std::size_t>(mat_.size())};
    }

    std::vector<vbx_float> ContractWith(const TensorField& other) const override {
        auto od = other.Data();
        if (static_cast<int>(od.size()) != mat_.cols())
            throw std::runtime_error("RankAdaptiveTensor: contraction dim mismatch");
        Eigen::Map<const Eigen::VectorXf> v(od.data(), od.size());
        Eigen::VectorXf r = mat_ * v;
        return {r.data(), r.data() + r.size()};
    }

    void AdaptToDimension(vbx_dim_t new_dim) override {
        if (new_dim == dim_) return;
        Eigen::MatrixXf nm = Eigen::MatrixXf::Identity(new_dim, new_dim);
        vbx_dim_t copy = std::min(new_dim, dim_);
        nm.topLeftCorner(copy, copy) = mat_.topLeftCorner(copy, copy);
        mat_ = std::move(nm);
        dim_ = new_dim;
    }

    void AddScaled(const TensorField& other, vbx_float scale) override {
        auto od = other.Data();
        for (std::size_t i = 0; i < od.size() && i < static_cast<std::size_t>(mat_.size()); ++i)
            mat_.data()[i] += scale * od[i];
    }

    std::unique_ptr<TensorField> Clone() const override {
        auto c = std::make_unique<RankAdaptiveTensor>(region_, dim_);
        c->mat_ = mat_;
        return c;
    }

private:
    vbx_region_id   region_;
    vbx_dim_t       dim_;
    Eigen::MatrixXf mat_;
};

std::unique_ptr<TensorField> MakeRankAdaptiveTensor(vbx_region_id r, vbx_dim_t dim) {
    return std::make_unique<RankAdaptiveTensor>(r, dim);
}

} // namespace vbx
