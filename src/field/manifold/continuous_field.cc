#include "src/field/field_impls.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

namespace vbx {

class ContinuousField : public LatentField {
public:
    explicit ContinuousField(std::size_t dim, std::size_t grid_pts = 1)
        : dim_(dim), grid_pts_(grid_pts), values_(Eigen::VectorXf::Zero(dim * grid_pts)) {}

    std::size_t Dim()  const override { return dim_; }
    std::size_t Size() const override { return static_cast<std::size_t>(values_.size()); }

    vbx_float Sample(Span<const vbx_float> coords) const override {
        if (grid_pts_ == 1) return values_(0);
        float t   = std::max(0.0f, std::min(1.0f, coords[0]));
        float idx = t * static_cast<float>(grid_pts_ - 1);
        int   lo  = static_cast<int>(idx);
        int   hi  = std::min(lo + 1, static_cast<int>(grid_pts_) - 1);
        float fr  = idx - lo;
        return (1.0f - fr) * values_(lo * static_cast<int>(dim_))
                           + fr  * values_(hi * static_cast<int>(dim_));
    }

    void Embed(Span<const vbx_float> vec) override {
        std::size_t n = std::min(vec.size(), static_cast<std::size_t>(values_.size()));
        for (std::size_t i = 0; i < n; ++i) values_(static_cast<int>(i)) = vec[i];
    }

    void Interpolate(Span<const vbx_float>, Span<vbx_float> out) const override {
        for (std::size_t i = 0; i < out.size() && i < dim_; ++i)
            out[i] = values_(static_cast<int>(i));
    }

    std::vector<vbx_float> Params() const override {
        return {values_.data(), values_.data() + values_.size()};
    }

    void SetParams(Span<const vbx_float> p) override {
        for (std::size_t i = 0; i < p.size() && i < static_cast<std::size_t>(values_.size()); ++i)
            values_(static_cast<int>(i)) = p[i];
    }

    void AddScaled(const LatentField& other, vbx_float scale) override {
        auto op = other.Params();
        for (std::size_t i = 0; i < op.size() && i < static_cast<std::size_t>(values_.size()); ++i)
            values_(static_cast<int>(i)) += scale * op[i];
    }

    void Scale(vbx_float s) override { values_ *= s; }

    std::unique_ptr<LatentField> Clone() const override {
        auto c = std::make_unique<ContinuousField>(dim_, grid_pts_);
        c->values_ = values_;
        return c;
    }

private:
    std::size_t     dim_;
    std::size_t     grid_pts_;
    Eigen::VectorXf values_;
};

std::unique_ptr<LatentField> MakeContinuousField(std::size_t dim, std::size_t grid_pts) {
    return std::make_unique<ContinuousField>(dim, grid_pts);
}

} // namespace vbx
