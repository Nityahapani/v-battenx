#pragma once

#include "version.h"
#include "context.h"
#include <string>

namespace vbx {

class GlobalConfig {
public:
    static GlobalConfig& Instance() {
        static GlobalConfig cfg;
        return cfg;
    }

    DeviceContext ctx;

    std::string version()     const { return VBATTENX_VERSION_STRING; }
    int         abi_version() const { return VBATTENX_ABI_VERSION; }

    void SetNumThreads(int n) { ctx.num_threads = n; }
    void SetGpuId(int id)     { ctx.gpu_id = id; }
    void SetLogLevel(LogLevel l) { ctx.log_level = l; }

private:
    GlobalConfig() = default;
};

} // namespace vbx
