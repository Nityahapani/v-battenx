#pragma once

#include "data.h"
#include "json.h"
#include <string>
#include <vector>
#include <stdexcept>

namespace vbx {

class PhysicsSpec {
public:
    PhysicsSpec& SetPde(PDETypeTag type, float diffusivity = 1.0f,
                        float dt = 0.01f, float viscosity = 1e-3f) {
        meta_.pde_type        = type;
        meta_.pde_diffusivity = diffusivity;
        meta_.pde_dt          = dt;
        meta_.pde_viscosity   = viscosity;
        return *this;
    }

    PhysicsSpec& AddSymmetry(SymmetryGroup g) {
        meta_.symmetry_groups.push_back(g);
        return *this;
    }

    PhysicsSpec& Conserve(const std::string& quantity) {
        meta_.conserved_quantities.push_back(quantity);
        return *this;
    }

    PhysicsSpec& SetBoundary(const std::string& region,
                              BCType type, float value = 0.0f, float flux = 0.0f) {
        meta_.boundary_conditions[region] = {type, value, flux};
        return *this;
    }

    PhysicsSpec& SetGrid(int nx, int ny, float resolution = 1.0f) {
        meta_.grid_nx             = nx;
        meta_.grid_ny             = ny;
        meta_.spatial_resolution  = resolution;
        return *this;
    }

    const PhysicalMetaInfo& Meta() const { return meta_; }

    JsonValue ToJson() const;
    static PhysicsSpec FromJson(const JsonValue& j);

private:
    static std::string PdeTagToStr(PDETypeTag t);
    static PDETypeTag  StrToPdeTag(const std::string& s);
    static std::string SymGroupToStr(SymmetryGroup g);
    static SymmetryGroup StrToSymGroup(const std::string& s);
    static std::string BcTypeToStr(BCType t);
    static BCType      StrToBcType(const std::string& s);

    PhysicalMetaInfo meta_;
};

} // namespace vbx
