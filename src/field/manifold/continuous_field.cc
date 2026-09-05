#include "vbatten_x/field.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>

namespace vbx {

class ContinuousField : public LatentField {
public:
    explicit ContinuousField(std::size_t dim, std::size_t grid_pts = 1)
        : dim_(dim), grid_pts_(grid_pts)
    {
        values_ = Eigen::VectorXf::Zero(dim_ * grid_pts_);
    }

    std::size_t Dim()  const override { return dim_; }
    std::size_t Size() const override { return values_.size(); }

    vbx_float Sample(Span<const vbx_float> coords) const override {
        if (grid_pts_ == 1) return values_(0);
        float t = std::max(0.0f, std::min(1.0f, coords[0]));
        float idx_f = t * (grid_pts_ - 1);
        int   lo    = static_cast<int>(idx_f);
        int   hi    = std::min(lo + 1, (int)grid_pts_ - 1);
        float frac  = idx_f - lo;
        return (1.0f - frac) * values_(lo * dim_) + frac * values_(hi * dim_);
    }

    void Embed(Span<const vbx_float> vec) override {
        std::size_t n = std::min(vec.size(), (std::size_t)values_.size());
        for (std::size_t i = 0; i < n; ++i) values_(i) = vec[i];
    }

    void Interpolate(Span<const vbx_float> coords, Span<vbx_float> out) const override {
        for (std::size_t i = 0; i < out.size() && i < dim_; ++i)
            out[i] = values_(i);
    }

    std::vector<vbx_float> Params() const override {
        return std::vector<vbx_float>(values_.data(), values_.data() + values_.size());
    }

    void SetParams(Span<const vbx_float> p) override {
        for (std::size_t i = 0; i < p.size() && i < (std::size_t)values_.size(); ++i)
            values_(i) = p[i];
    }

    void AddScaled(const LatentField& other, vbx_float scale) override {
        auto op = other.Params();
        for (std::size_t i = 0; i < op.size() && i < (std::size_t)values_.size(); ++i)
            values_(i) += scale * op[i];
    }

    void Scale(vbx_float s) override { values_ *= s; }

    std::unique_ptr<LatentField> Clone() const override {
        auto c = std::make_unique<ContinuousField>(dim_, grid_pts_);
        c->values_ = values_;
        return c;
    }

    const Eigen::VectorXf& EigenVec() const { return values_; }
    Eigen::VectorXf&       EigenVec()       { return values_; }

private:
    std::size_t     dim_;
    std::size_t     grid_pts_;
    Eigen::VectorXf values_;
};

std::unique_ptr<LatentField> MakeContinuousField(std::size_t dim, std::size_t grid_pts) {
    return std::make_unique<ContinuousField>(dim, grid_pts);
}

} // namespace vbx
