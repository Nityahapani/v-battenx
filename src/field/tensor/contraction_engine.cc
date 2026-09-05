#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>

namespace vbx {

class Rank2Tensor : public TensorField {
public:
    Rank2Tensor(vbx_region_id region, vbx_dim_t rows, vbx_dim_t cols)
        : region_(region), mat_(Eigen::MatrixXf::Zero(rows, cols)) {}

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
            throw std::runtime_error("Contraction dimension mismatch");
        Eigen::Map<const Eigen::VectorXf> vec(od.data(), od.size());
        Eigen::VectorXf result = mat_ * vec;
        return {result.data(), result.data() + result.size()};
    }

    void AdaptToDimension(vbx_dim_t new_dim) override {
        if (new_dim == mat_.rows()) return;
        Eigen::MatrixXf nm = Eigen::MatrixXf::Zero(new_dim, new_dim);
        vbx_dim_t cr = std::min(new_dim, static_cast<vbx_dim_t>(mat_.rows()));
        vbx_dim_t cc = std::min(new_dim, static_cast<vbx_dim_t>(mat_.cols()));
        nm.topLeftCorner(cr, cc) = mat_.topLeftCorner(cr, cc);
        mat_ = std::move(nm);
    }

    void AddScaled(const TensorField& other, vbx_float scale) override {
        auto od = other.Data();
        for (std::size_t i = 0; i < od.size() && i < static_cast<std::size_t>(mat_.size()); ++i)
            mat_.data()[i] += scale * od[i];
    }

    std::unique_ptr<TensorField> Clone() const override {
        auto c = std::make_unique<Rank2Tensor>(region_, mat_.rows(), mat_.cols());
        c->mat_ = mat_;
        return c;
    }

private:
    vbx_region_id   region_;
    Eigen::MatrixXf mat_;
};

std::unique_ptr<TensorField> MakeRank2Tensor(vbx_region_id r, vbx_dim_t rows, vbx_dim_t cols) {
    return std::make_unique<Rank2Tensor>(r, rows, cols);
}

} // namespace vbx
