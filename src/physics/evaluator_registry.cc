#include "vbatten_x/physics_evaluator.h"
#include "vbatten_x/data.h"
#include "src/physics/pde/heat_equation.cc"
#include "src/physics/pde/navier_stokes.cc"
#include "src/physics/pde/custom_pde.cc"

namespace vbx {

std::unique_ptr<PhysicsEvaluator> MakeEvaluatorFromMeta(const PhysicalMetaInfo& meta) {
    switch (meta.pde_type) {
        case PDETypeTag::Heat:
            return MakeHeatEvaluator(
                meta.pde_diffusivity, meta.pde_dt,
                meta.grid_nx, meta.grid_ny, meta.spatial_resolution);
        case PDETypeTag::NavierStokes:
            return MakeNavierStokesEvaluator(
                meta.pde_viscosity, meta.pde_dt,
                meta.grid_nx, meta.grid_ny, meta.spatial_resolution);
        default:
            return MakeNullEvaluator();
    }
}

} // namespace vbx
