#pragma once

#include "base.h"
#include "data.h"
#include <memory>

namespace vbx {

struct FieldState;
struct EncoderParam;

class PhysicsEncoder {
public:
    virtual ~PhysicsEncoder() = default;
    virtual FieldState Encode(const PhysicalDataset& ds) = 0;
    virtual void       UpdateParams(const FieldState& grad, vbx_float lr) = 0;
};

} // namespace vbx
