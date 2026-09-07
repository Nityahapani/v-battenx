#pragma once

#ifdef VBATTENX_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <string>

namespace vbx {

class GpuMemoryPool {
public:
    GpuMemoryPool(std::size_t pool_bytes)
        : pool_bytes_(pool_bytes), offset_(0) {
        cudaMalloc(&base_, pool_bytes_);
    }

    ~GpuMemoryPool() {
        if (base_) cudaFree(base_);
    }

    void* Alloc(std::size_t bytes) {
        bytes = (bytes + 255) & ~255ull;  // 256-byte alignment
        if (offset_ + bytes > pool_bytes_)
            throw std::runtime_error("GpuMemoryPool: out of memory");
        void* ptr = static_cast<uint8_t*>(base_) + offset_;
        offset_  += bytes;
        return ptr;
    }

    void Reset() { offset_ = 0; }

    std::size_t Used()      const { return offset_; }
    std::size_t Remaining() const { return pool_bytes_ - offset_; }

private:
    void*       base_       = nullptr;
    std::size_t pool_bytes_;
    std::size_t offset_;
};

} // namespace vbx
#endif // VBATTENX_CUDA
