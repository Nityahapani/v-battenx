#pragma once

#include <cstdint>
#include <string>

namespace vbx {

enum class LogLevel : int { Silent = 0, Warning = 1, Info = 2, Debug = 3 };

struct DeviceContext {
    int         num_threads = 1;
    int         gpu_id      = -1;
    uint64_t    rng_seed    = 42;
    LogLevel    log_level   = LogLevel::Info;

    static DeviceContext Default() { return {}; }
};

} // namespace vbx
