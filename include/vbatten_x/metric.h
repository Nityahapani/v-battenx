#pragma once

#include "base.h"
#include "span.h"
#include <string>
#include <memory>

namespace vbx {

class PhysicsMetric {
public:
    virtual ~PhysicsMetric() = default;
    virtual vbx_float Eval(Span<const vbx_float> pred,
                           Span<const vbx_float> label) const = 0;
    virtual std::string Name()       const = 0;
    virtual bool        Maximize()   const { return false; }
};

std::unique_ptr<PhysicsMetric> MakeRmseMetric();
std::unique_ptr<PhysicsMetric> MakeAucMetric();

} // namespace vbx
