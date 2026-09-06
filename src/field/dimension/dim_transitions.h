#pragma once
#include "vbatten_x/base.h"
#include <vector>
#include <cstdint>

namespace vbx {
std::vector<float> Expand1dTo2d(const std::vector<float>& params, uint64_t seed = 42);
std::vector<float> Expand2dTo3d(const std::vector<float>& params, uint64_t seed = 42);
std::vector<float> Collapse3dTo2d(const std::vector<float>& params);
std::vector<float> Collapse2dTo1d(const std::vector<float>& params);
std::vector<float> LocalDimChange(const std::vector<float>& params,
                                   vbx_dim_t from_d, vbx_dim_t to_d,
                                   uint64_t seed = 42);
}
