#include "vbatten_x/tensor_field.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <cstring>

namespace vbx {

class Rank2Tensor : public TensorField {
public:
    Rank2Tensor(vbx_region_id region, vbx_dim_t rows, vbx_dim_t cols)
        : region_(region), mat_(Eigen::MatrixXf::Zero(rows, cols)) {}

    int           Rank()    const override { return 2; }
    std::size_t   Size()    const override { return mat_.size(); }
    vbx_region_id Region()  const override { return region_; }

    Span<const vbx_float> Data() const override {
        return {mat_.data(), static_cast<std::size_t>(mat_.size())};
    }
    Span<vbx_float> Data() override {
        return {mat_.data(), static_cast<std::size_t>(mat_.size())};
    }

    std::vector<vbx_float> ContractWith(const TensorField& other) const override {
        auto od = other.Data();
        vbx_dim_t rows = mat_.rows();
        vbx_dim_t cols = mat_.cols();
        if ((vbx_dim_t)od.size() != cols)
            throw std::runtime_error("Contraction dimension mismatch");

        Eigen::Map<const Eigen::VectorXf> vec(od.data(), od.size());
        Eigen::VectorXf result = mat_ * vec;
        return std::vector<vbx_float>(result.data(), result.data() + result.size());
    }

    void AdaptToDimension(vbx_dim_t new_dim) override {
        vbx_dim_t old_rows = mat_.rows();
        vbx_dim_t old_cols = mat_.cols();
        if (new_dim == old_rows) return;
        Eigen::MatrixXf nm = Eigen::MatrixXf::Zero(new_dim, new_dim);
        vbx_dim_t copy_r = std::min(new_dim, old_rows);
        vbx_dim_t copy_c = std::min(new_dim, old_cols);
        nm.topLeftCorner(copy_r, copy_c) = mat_.topLeftCorner(copy_r, copy_c);
        mat_ = std::move(nm);
    }

    void AddScaled(const TensorField& other, vbx_float scale) override {
        auto od = other.Data();
        for (std::size_t i = 0; i < od.size() && i < (std::size_t)mat_.size(); ++i)
            mat_.data()[i] += scale * od[i];
    }

    std::unique_ptr<TensorField> Clone() const override {
        auto c = std::make_unique<Rank2Tensor>(region_, mat_.rows(), mat_.cols());
        c->mat_ = mat_;
        return c;
    }

    Eigen::MatrixXf& Mat()       { return mat_; }
    const Eigen::MatrixXf& Mat() const { return mat_; }

private:
    vbx_region_id   region_;
    Eigen::MatrixXf mat_;
};

std::unique_ptr<TensorField> MakeRank2Tensor(vbx_region_id r, vbx_dim_t rows, vbx_dim_t cols) {
    return std::make_unique<Rank2Tensor>(r, rows, cols);
}

} // namespace vbx
