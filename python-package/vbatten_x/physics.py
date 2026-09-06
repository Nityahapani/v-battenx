from __future__ import annotations

import json
from enum import Enum
from typing import Optional, Dict, Any


class PDEType(str, Enum):
    NONE          = "none"
    HEAT          = "heat"
    WAVE          = "wave"
    NAVIER_STOKES = "navier_stokes"
    POISSON       = "poisson"
    CUSTOM        = "custom"


class SymmetryGroup(str, Enum):
    ROTATION_2D  = "rotation_2d"
    ROTATION_3D  = "rotation_3d"
    REFLECTION   = "reflection"
    PERMUTATION  = "permutation"
    TRANSLATION  = "translation"


class BCType(str, Enum):
    DIRICHLET = "dirichlet"
    NEUMANN   = "neumann"
    PERIODIC  = "periodic"


class PhysicsSpec:
    def __init__(self):
        self._spec: Dict[str, Any] = {
            "pde_type":             PDEType.NONE.value,
            "pde_diffusivity":      1.0,
            "pde_dt":               0.01,
            "pde_viscosity":        1e-3,
            "grid_nx":              16,
            "grid_ny":              16,
            "spatial_resolution":   1.0,
            "symmetry_groups":      [],
            "conserved_quantities": [],
            "boundary_conditions":  {},
        }

    def pde(self,
            pde_type: PDEType,
            diffusivity: float = 1.0,
            dt: float = 0.01,
            viscosity: float = 1e-3) -> "PhysicsSpec":
        self._spec["pde_type"]        = pde_type.value if isinstance(pde_type, PDEType) else pde_type
        self._spec["pde_diffusivity"] = diffusivity
        self._spec["pde_dt"]          = dt
        self._spec["pde_viscosity"]   = viscosity
        return self

    def symmetry(self, group: SymmetryGroup) -> "PhysicsSpec":
        v = group.value if isinstance(group, SymmetryGroup) else group
        if v not in self._spec["symmetry_groups"]:
            self._spec["symmetry_groups"].append(v)
        return self

    def conserve(self, quantity: str) -> "PhysicsSpec":
        if quantity not in self._spec["conserved_quantities"]:
            self._spec["conserved_quantities"].append(quantity)
        return self

    def boundary(self,
                 region: str,
                 bc_type: BCType,
                 value: float = 0.0,
                 flux: float = 0.0) -> "PhysicsSpec":
        t = bc_type.value if isinstance(bc_type, BCType) else bc_type
        self._spec["boundary_conditions"][region] = {
            "type": t, "value": value, "flux": flux
        }
        return self

    def grid(self, nx: int, ny: int, resolution: float = 1.0) -> "PhysicsSpec":
        self._spec["grid_nx"]            = nx
        self._spec["grid_ny"]            = ny
        self._spec["spatial_resolution"] = resolution
        return self

    def to_json(self) -> str:
        return json.dumps(self._spec)

    @classmethod
    def from_json(cls, s: str) -> "PhysicsSpec":
        spec = cls()
        spec._spec = json.loads(s)
        return spec

    def __repr__(self) -> str:
        return f"PhysicsSpec(pde={self._spec['pde_type']}, symmetries={self._spec['symmetry_groups']})"
