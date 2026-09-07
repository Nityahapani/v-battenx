#pragma once

#ifdef VBATTENX_CUDA
#include <cuda_runtime.h>

#define VBATTENX_CUDA_CHECK(expr)                                          \
    do {                                                                    \
        cudaError_t _err = (expr);                                          \
        if (_err != cudaSuccess)                                            \
            throw std::runtime_error(std::string("CUDA error: ")           \
                + cudaGetErrorString(_err)                                  \
                + " in " __FILE__ ":" + std::to_string(__LINE__));         \
    } while (0)

inline int NumGpus() {
    int n = 0;
    cudaGetDeviceCount(&n);
    return n;
}

struct CudaStream {
    cudaStream_t stream = nullptr;
    CudaStream()  { VBATTENX_CUDA_CHECK(cudaStreamCreate(&stream)); }
    ~CudaStream() { if (stream) cudaStreamDestroy(stream); }
    CudaStream(const CudaStream&)            = delete;
    CudaStream& operator=(const CudaStream&) = delete;
};

#else

#define VBATTENX_CUDA_CHECK(expr)  ((void)(expr))

inline int NumGpus() { return 0; }

#endif // VBATTENX_CUDA
