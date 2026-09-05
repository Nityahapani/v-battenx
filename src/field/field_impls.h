#pragma once
#include "vbatten_x/field.h"
#include "vbatten_x/topology.h"
#include "vbatten_x/dimension.h"
#include "vbatten_x/tensor_field.h"
#include <memory>

namespace vbx {
std::unique_ptr<LatentField>   MakeContinuousField(std::size_t dim, std::size_t grid_pts = 1);
std::unique_ptr<FieldTopology> MakeRegionGraph();
std::unique_ptr<DimensionMap>  MakeUniformDim(vbx_dim_t dim, std::size_t num_regions, float budget);
std::unique_ptr<TensorField>   MakeRank2Tensor(vbx_region_id r, vbx_dim_t rows, vbx_dim_t cols);
}
