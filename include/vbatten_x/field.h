#pragma once

#include "base.h"
#include "span.h"
#include <memory>
#include <vector>

namespace vbx {

class LatentField {
public:
    virtual ~LatentField() = default;

    virtual std::size_t Dim()  const = 0;
    virtual std::size_t Size() const = 0;

    virtual vbx_float Sample(Span<const vbx_float> coords) const = 0;
    virtual void      Embed(Span<const vbx_float> vec) = 0;
    virtual void      Interpolate(Span<const vbx_float> coords,
                                  Span<vbx_float> out) const = 0;

    virtual std::vector<vbx_float> Params() const = 0;
    virtual void SetParams(Span<const vbx_float> params) = 0;

    virtual std::unique_ptr<LatentField> Clone() const = 0;

    virtual void AddScaled(const LatentField& other, vbx_float scale) = 0;
    virtual void Scale(vbx_float s) = 0;
};

} // namespace vbx
