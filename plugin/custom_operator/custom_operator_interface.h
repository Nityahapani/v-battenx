#pragma once

#include "vbatten_x/topological_operator.h"
#include <memory>
#include <string>

// Plugin interface for user-defined TopologicalOperators.
// Implement this interface in a shared library and register it
// via VBATTENX_REGISTER_OPERATOR.

namespace vbx {

using OperatorFactoryFn = std::unique_ptr<TopologicalOperator>(*)();

struct PluginOperatorDescriptor {
    const char*       name;
    const char*       version;
    const char*       description;
    OperatorFactoryFn factory;
};

} // namespace vbx

#define VBATTENX_REGISTER_OPERATOR(name_, factory_fn_, desc_)        \
    extern "C" {                                                       \
    vbx::PluginOperatorDescriptor vbattenx_plugin_descriptor = {      \
        name_, "1.0", desc_, factory_fn_                              \
    };                                                                 \
    }
