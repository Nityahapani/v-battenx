#pragma once

#ifdef VBATTENX_CUDA
#include <cuda_runtime.h>
#include "vbatten_x/base.h"
#include "src/field/field_state.h"
#include <vector>
#include <stdexcept>

namespace vbx {

struct GpuFieldState {
    vbx_float* F_data      = nullptr;
    std::size_t F_size     = 0;
    vbx_float* T_data      = nullptr;
    std::size_t T_size     = 0;
    int         num_regions = 0;
    cudaStream_t stream    = nullptr;

    static GpuFieldState ToDevice(const FieldState& s) {
        GpuFieldState g;
        auto params = s.F->Params();
        g.F_size    = params.size();
        cudaMalloc(&g.F_data, g.F_size * sizeof(vbx_float));
        cudaMemcpy(g.F_data, params.data(),
                   g.F_size * sizeof(vbx_float), cudaMemcpyHostToDevice);
        auto tdata = s.T->Data();
        g.T_size   = tdata.size();
        cudaMalloc(&g.T_data, g.T_size * sizeof(vbx_float));
        cudaMemcpy(g.T_data, tdata.data(),
                   g.T_size * sizeof(vbx_float), cudaMemcpyHostToDevice);
        g.num_regions = s.K ? static_cast<int>(s.K->NumRegions()) : 1;
        return g;
    }

    std::vector<vbx_float> FToHost() const {
        std::vector<vbx_float> out(F_size);
        cudaMemcpy(out.data(), F_data,
                   F_size * sizeof(vbx_float), cudaMemcpyDeviceToHost);
        return out;
    }

    void Free() {
        if (F_data) { cudaFree(F_data); F_data = nullptr; }
        if (T_data) { cudaFree(T_data); T_data = nullptr; }
    }
};

} // namespace vbx
#endif // VBATTENX_CUDA
