# VBattenX Physics Spec builder for R

#' Create a PhysicsSpec builder
#' @return vbx.PhysicsSpec object
#' @export
vbx.physics.spec <- function() {
  spec <- list(
    pde_type             = "none",
    pde_diffusivity      = 1.0,
    pde_dt               = 0.01,
    pde_viscosity        = 1e-3,
    grid_nx              = 16L,
    grid_ny              = 16L,
    spatial_resolution   = 1.0,
    symmetry_groups      = list(),
    conserved_quantities = list(),
    boundary_conditions  = list()
  )
  structure(spec, class = "vbx.PhysicsSpec")
}

#' Set the PDE type on a PhysicsSpec
#' @export
vbx.pde <- function(spec, type, diffusivity = 1.0, dt = 0.01, viscosity = 1e-3) {
  stopifnot(inherits(spec, "vbx.PhysicsSpec"))
  spec$pde_type        <- type
  spec$pde_diffusivity <- diffusivity
  spec$pde_dt          <- dt
  spec$pde_viscosity   <- viscosity
  spec
}

#' Add a symmetry group
#' @export
vbx.symmetry <- function(spec, group) {
  stopifnot(inherits(spec, "vbx.PhysicsSpec"))
  if (!group %in% spec$symmetry_groups)
    spec$symmetry_groups <- c(spec$symmetry_groups, group)
  spec
}

#' Declare a conserved quantity
#' @export
vbx.conserve <- function(spec, quantity) {
  stopifnot(inherits(spec, "vbx.PhysicsSpec"))
  spec$conserved_quantities <- c(spec$conserved_quantities, quantity)
  spec
}

#' Set boundary condition
#' @export
vbx.boundary <- function(spec, region, type, value = 0.0, flux = 0.0) {
  stopifnot(inherits(spec, "vbx.PhysicsSpec"))
  spec$boundary_conditions[[region]] <- list(type = type, value = value, flux = flux)
  spec
}

#' Set grid dimensions
#' @export
vbx.grid <- function(spec, nx, ny, resolution = 1.0) {
  stopifnot(inherits(spec, "vbx.PhysicsSpec"))
  spec$grid_nx             <- as.integer(nx)
  spec$grid_ny             <- as.integer(ny)
  spec$spatial_resolution  <- resolution
  spec
}

#' Convert PhysicsSpec to JSON string
#' @export
vbx.spec.to_json <- function(spec) {
  jsonlite::toJSON(unclass(spec), auto_unbox = TRUE)
}

#' @export
print.vbx.PhysicsSpec <- function(x, ...) {
  cat("VBattenX PhysicsSpec\n")
  cat("  PDE:       ", x$pde_type, "\n")
  cat("  Symmetries:", paste(x$symmetry_groups, collapse = ", "), "\n")
  cat("  Conserved: ", paste(x$conserved_quantities, collapse = ", "), "\n")
  invisible(x)
}
