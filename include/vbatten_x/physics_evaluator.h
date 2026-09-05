#pragma once

#include "base.h"
#include <vector>
#include <memory>

namespace vbx {

class PhysicalDataset;
struct FieldState;

struct ResidualInfo {
    std::vector<vbx_float> pde_residual;
    std::vector<vbx_float> prediction_residual;
    std::vector<vbx_float> constraint_violation;

    vbx_float MeanPde()        const;
    vbx_float MeanPrediction() const;
};

class PhysicsEvaluator {
public:
    virtual ~PhysicsEvaluator() = default;
    virtual ResidualInfo Eval(const FieldState& state,
                              const PhysicalDataset& ds) const = 0;
};

std::unique_ptr<PhysicsEvaluator> MakeNullEvaluator();

} // namespace vbx
