#pragma once

#include "base.h"
#include "span.h"
#include <vector>
#include <memory>
#include <string>

namespace vbx {

struct GradPair {
    std::vector<vbx_float> g;
    std::vector<vbx_float> h;
};

class Objective {
public:
    virtual ~Objective() = default;
    virtual GradPair  GetGradients(Span<const vbx_float> pred,
                                   Span<const vbx_float> label) const = 0;
    virtual vbx_float Loss(Span<const vbx_float> pred,
                           Span<const vbx_float> label) const = 0;
    virtual std::string Name() const = 0;
};

std::unique_ptr<Objective> MakeRegressionObjective();
std::unique_ptr<Objective> MakeClassificationObjective();

} // namespace vbx
