#pragma once

#include "base.h"
#include "feature_map.h"
#include "span.h"
#include <string>
#include <vector>
#include <memory>

namespace vbx {

enum class PDETypeTag { None, Heat, Wave, NavierStokes, Poisson, Custom };

struct PhysicalMetaInfo {
    PDETypeTag        pde_type   = PDETypeTag::None;
    std::vector<std::string> symmetry_groups;
    std::string       spatial_unit;
    double            spatial_resolution = 1.0;
};

class PhysicalDataset {
public:
    virtual ~PhysicalDataset() = default;

    virtual vbx_index  NumRows() const = 0;
    virtual vbx_index  NumCols() const = 0;

    virtual Span<const vbx_float> GetRow(vbx_index row) const = 0;
    virtual Span<const vbx_float> GetCol(vbx_index col) const = 0;

    virtual const vbx_float* RawData()   const = 0;
    virtual const vbx_float* Labels()    const = 0;
    virtual bool              HasLabels() const = 0;

    virtual const FeatureMap&      Features() const = 0;
    virtual const PhysicalMetaInfo& Meta()    const = 0;
};

} // namespace vbx
