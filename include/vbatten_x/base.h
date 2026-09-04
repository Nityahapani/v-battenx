#pragma once

#include <cstdint>
#include <cstddef>

#ifdef VBATTENX_USE_FLOAT64
  using vbx_float = double;
#else
  using vbx_float = float;
#endif

using vbx_index = std::int64_t;
using vbx_dim_t = std::int32_t;
using vbx_region_id = std::uint32_t;

#ifdef VBATTENX_CUDA
  #define VBX_HOST_DEVICE __host__ __device__
  #define VBX_DEVICE      __device__
#else
  #define VBX_HOST_DEVICE
  #define VBX_DEVICE
#endif

constexpr vbx_region_id kInvalidRegion = 0xFFFFFFFF;
