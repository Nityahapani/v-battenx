#pragma once

#include "base.h"
#include "span.h"
#include <memory>
#include <vector>

namespace vbx {

class TensorField {
public:
    virtual ~TensorField() = default;

    virtual int         Rank()    const = 0;
    virtual std::size_t Size()    const = 0;
    virtual vbx_region_id Region() const = 0;

    virtual Span<const vbx_float> Data() const = 0;
    virtual Span<vbx_float>       Data() = 0;

    virtual std::vector<vbx_float> ContractWith(const TensorField& other) const = 0;
    virtual void AdaptToDimension(vbx_dim_t new_dim) = 0;

    virtual std::unique_ptr<TensorField> Clone() const = 0;
    virtual void AddScaled(const TensorField& other, vbx_float scale) = 0;
};

} // namespace vbx
