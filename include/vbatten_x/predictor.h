#pragma once

#include "base.h"
#include <vector>
#include <memory>

namespace vbx {

class PhysicalDataset;
struct FieldState;

class FieldPredictor {
public:
    virtual ~FieldPredictor() = default;
    virtual std::vector<vbx_float> Predict(const PhysicalDataset& ds,
                                            const FieldState& stage) const = 0;
};

std::unique_ptr<FieldPredictor> MakeCpuPredictor();

} // namespace vbx
