#pragma once

#include "base.h"
#include <string>
#include <memory>

namespace vbx {

class VBattenLearner;

class Model {
public:
    virtual ~Model() = default;
    virtual void Save(const std::string& path) const = 0;
    virtual void Load(const std::string& path)       = 0;
    virtual std::string FormatVersion() const        = 0;
};

} // namespace vbx
