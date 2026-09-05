#pragma once

#include "base.h"
#include "parameter.h"
#include "span.h"
#include <vector>
#include <memory>
#include <functional>
#include <string>

namespace vbx {

class PhysicalDataset;

struct EvalResult {
    std::string metric_name;
    vbx_float   value;
    int         iteration;
};

using EvalCallback = std::function<void(int iter, const EvalResult&)>;

class VBattenLearner {
public:
    virtual ~VBattenLearner() = default;

    virtual void Train(const PhysicalDataset& ds,
                       int num_iters,
                       EvalCallback cb = nullptr) = 0;

    virtual std::vector<vbx_float> Predict(const PhysicalDataset& ds) const = 0;

    virtual void Save(const std::string& path) const = 0;
    virtual void Load(const std::string& path) = 0;

    virtual int   NumStages()  const = 0;
    virtual float TrainLoss()  const = 0;
};

std::unique_ptr<VBattenLearner> MakeLearner(VBXParameter params);

} // namespace vbx
