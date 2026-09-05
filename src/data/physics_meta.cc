#include "vbatten_x/physics_spec.h"
#include <stdexcept>

namespace vbx {

std::string PhysicsSpec::PdeTagToStr(PDETypeTag t) {
    switch (t) {
        case PDETypeTag::None:         return "none";
        case PDETypeTag::Heat:         return "heat";
        case PDETypeTag::Wave:         return "wave";
        case PDETypeTag::NavierStokes: return "navier_stokes";
        case PDETypeTag::Poisson:      return "poisson";
        case PDETypeTag::Custom:       return "custom";
    }
    return "none";
}

PDETypeTag PhysicsSpec::StrToPdeTag(const std::string& s) {
    if (s == "heat")          return PDETypeTag::Heat;
    if (s == "wave")          return PDETypeTag::Wave;
    if (s == "navier_stokes") return PDETypeTag::NavierStokes;
    if (s == "poisson")       return PDETypeTag::Poisson;
    if (s == "custom")        return PDETypeTag::Custom;
    return PDETypeTag::None;
}

std::string PhysicsSpec::SymGroupToStr(SymmetryGroup g) {
    switch (g) {
        case SymmetryGroup::Rotation2D:  return "rotation_2d";
        case SymmetryGroup::Rotation3D:  return "rotation_3d";
        case SymmetryGroup::Reflection:  return "reflection";
        case SymmetryGroup::Permutation: return "permutation";
        case SymmetryGroup::Translation: return "translation";
    }
    return "rotation_2d";
}

SymmetryGroup PhysicsSpec::StrToSymGroup(const std::string& s) {
    if (s == "rotation_3d")  return SymmetryGroup::Rotation3D;
    if (s == "reflection")   return SymmetryGroup::Reflection;
    if (s == "permutation")  return SymmetryGroup::Permutation;
    if (s == "translation")  return SymmetryGroup::Translation;
    return SymmetryGroup::Rotation2D;
}

std::string PhysicsSpec::BcTypeToStr(BCType t) {
    switch (t) {
        case BCType::Dirichlet: return "dirichlet";
        case BCType::Neumann:   return "neumann";
        case BCType::Periodic:  return "periodic";
    }
    return "dirichlet";
}

BCType PhysicsSpec::StrToBcType(const std::string& s) {
    if (s == "neumann")  return BCType::Neumann;
    if (s == "periodic") return BCType::Periodic;
    return BCType::Dirichlet;
}

JsonValue PhysicsSpec::ToJson() const {
    auto root = JsonValue::MakeObject();
    root.Set("pde_type",        JsonValue(PdeTagToStr(meta_.pde_type)));
    root.Set("pde_diffusivity", JsonValue(static_cast<double>(meta_.pde_diffusivity)));
    root.Set("pde_dt",          JsonValue(static_cast<double>(meta_.pde_dt)));
    root.Set("pde_viscosity",   JsonValue(static_cast<double>(meta_.pde_viscosity)));
    root.Set("grid_nx",         JsonValue(meta_.grid_nx));
    root.Set("grid_ny",         JsonValue(meta_.grid_ny));
    root.Set("spatial_resolution", JsonValue(static_cast<double>(meta_.spatial_resolution)));

    auto sym_arr = JsonValue::MakeArray();
    for (auto g : meta_.symmetry_groups)
        sym_arr.Append(JsonValue(SymGroupToStr(g)));
    root.Set("symmetry_groups", std::move(sym_arr));

    auto cons_arr = JsonValue::MakeArray();
    for (auto& q : meta_.conserved_quantities)
        cons_arr.Append(JsonValue(q));
    root.Set("conserved_quantities", std::move(cons_arr));

    auto bc_obj = JsonValue::MakeObject();
    for (auto& [region, bc] : meta_.boundary_conditions) {
        auto bc_entry = JsonValue::MakeObject();
        bc_entry.Set("type",  JsonValue(BcTypeToStr(bc.type)));
        bc_entry.Set("value", JsonValue(static_cast<double>(bc.value)));
        bc_entry.Set("flux",  JsonValue(static_cast<double>(bc.flux)));
        bc_obj.Set(region, std::move(bc_entry));
    }
    root.Set("boundary_conditions", std::move(bc_obj));
    return root;
}

PhysicsSpec PhysicsSpec::FromJson(const JsonValue& j) {
    PhysicsSpec s;
    s.meta_.pde_type        = StrToPdeTag(j["pde_type"].AsString());
    s.meta_.pde_diffusivity = static_cast<float>(j["pde_diffusivity"].AsDouble());
    s.meta_.pde_dt          = static_cast<float>(j["pde_dt"].AsDouble());
    s.meta_.pde_viscosity   = static_cast<float>(j["pde_viscosity"].AsDouble());
    s.meta_.grid_nx         = j["grid_nx"].AsInt();
    s.meta_.grid_ny         = j["grid_ny"].AsInt();
    s.meta_.spatial_resolution = static_cast<float>(j["spatial_resolution"].AsDouble());

    for (std::size_t i = 0; i < j["symmetry_groups"].ArraySize(); ++i)
        s.meta_.symmetry_groups.push_back(StrToSymGroup(j["symmetry_groups"][i].AsString()));

    for (std::size_t i = 0; i < j["conserved_quantities"].ArraySize(); ++i)
        s.meta_.conserved_quantities.push_back(j["conserved_quantities"][i].AsString());

    for (auto& [region, bc_val] : j["boundary_conditions"].AsObject()) {
        BoundaryCondition bc;
        bc.type  = StrToBcType(bc_val["type"].AsString());
        bc.value = static_cast<float>(bc_val["value"].AsDouble());
        bc.flux  = static_cast<float>(bc_val["flux"].AsDouble());
        s.meta_.boundary_conditions[region] = bc;
    }
    return s;
}

} // namespace vbx
