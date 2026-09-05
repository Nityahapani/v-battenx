#pragma once

#include "base.h"
#include "feature_map.h"
#include "span.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace vbx {

enum class PDETypeTag {
    None, Heat, Wave, NavierStokes, Poisson, Custom
};

enum class SymmetryGroup {
    Rotation2D, Rotation3D, Reflection, Permutation, Translation
};

enum class BCType {
    Dirichlet, Neumann, Periodic
};

struct BoundaryCondition {
    BCType    type;
    float     value = 0.0f;
    float     flux  = 0.0f;
};

struct PhysicalMetaInfo {
    PDETypeTag   pde_type          = PDETypeTag::None;
    float        pde_diffusivity   = 1.0f;
    float        pde_dt            = 0.01f;
    float        pde_viscosity     = 1e-3f;

    std::vector<SymmetryGroup>   symmetry_groups;
    std::vector<std::string>     conserved_quantities;

    std::unordered_map<std::string, BoundaryCondition> boundary_conditions;

    std::string  spatial_unit;
    float        spatial_resolution = 1.0f;
    int          grid_nx = 16;
    int          grid_ny = 16;
};

class PhysicalDataset {
public:
    virtual ~PhysicalDataset() = default;

    virtual vbx_index  NumRows() const = 0;
    virtual vbx_index  NumCols() const = 0;

    virtual Span<const vbx_float> GetRow(vbx_index row) const = 0;
    virtual Span<const vbx_float> GetCol(vbx_index col) const = 0;

    virtual const vbx_float* RawData()    const = 0;
    virtual const vbx_float* Labels()     const = 0;
    virtual bool              HasLabels()  const = 0;

    virtual const FeatureMap&       Features() const = 0;
    virtual const PhysicalMetaInfo& Meta()     const = 0;
};

} // namespace vbx
