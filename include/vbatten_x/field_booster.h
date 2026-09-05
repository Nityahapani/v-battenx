#pragma once

#include "base.h"
#include "span.h"
#include <memory>
#include <vector>

namespace vbx {

class PhysicalDataset;
struct FieldState;
struct GradPair;

struct FieldPrediction {
    std::vector<vbx_float> values;
};

class FieldBooster {
public:
    virtual ~FieldBooster() = default;
    virtual FieldState     DoBoost(const PhysicalDataset& ds, const GradPair& gp) = 0;
    virtual FieldPrediction PredictStage(const PhysicalDataset& ds,
                                         const FieldState& stage) const = 0;
};

} // namespace vbx
