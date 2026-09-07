#pragma once

#include "vbatten_x/physics_evaluator.h"
#include <memory>
#include <string>

namespace vbx {

using PdeFactoryFn = std::unique_ptr<PhysicsEvaluator>(*)();

struct PluginPdeDescriptor {
    const char*   name;
    const char*   version;
    const char*   description;
    PdeFactoryFn  factory;
};

} // namespace vbx

#define VBATTENX_REGISTER_PDE(name_, factory_fn_, desc_)       \
    extern "C" {                                                 \
    vbx::PluginPdeDescriptor vbattenx_pde_descriptor = {        \
        name_, "1.0", desc_, factory_fn_                        \
    };                                                           \
    }
