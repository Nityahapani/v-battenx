#pragma once
#include "vbatten_x/field.h"
#include "vbatten_x/topology.h"
#include "vbatten_x/dimension.h"
#include "vbatten_x/tensor_field.h"
#include <memory>
#include <vector>

namespace vbx {

struct FieldState {
    std::unique_ptr<LatentField>   F;
    std::unique_ptr<FieldTopology> K;
    std::unique_ptr<DimensionMap>  d;
    std::unique_ptr<TensorField>   T;
    std::vector<vbx_float>         bias;

    FieldState() = default;
    FieldState(FieldState&&) = default;
    FieldState& operator=(FieldState&&) = default;

    FieldState Clone() const;
    double     Diff(const FieldState& other) const;
};

} // namespace vbx
