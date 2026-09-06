import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import json
import pytest
from vbatten_x.physics import PhysicsSpec, PDEType, SymmetryGroup, BCType


def test_default_spec_is_none_pde():
    spec = PhysicsSpec()
    d = json.loads(spec.to_json())
    assert d["pde_type"] == "none"


def test_pde_builder():
    spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.01, dt=0.005)
    d = json.loads(spec.to_json())
    assert d["pde_type"]        == "heat"
    assert abs(d["pde_diffusivity"] - 0.01) < 1e-9
    assert abs(d["pde_dt"]          - 0.005) < 1e-9


def test_symmetry_builder():
    spec = PhysicsSpec().symmetry(SymmetryGroup.ROTATION_2D).symmetry(SymmetryGroup.REFLECTION)
    d = json.loads(spec.to_json())
    assert "rotation_2d" in d["symmetry_groups"]
    assert "reflection"  in d["symmetry_groups"]


def test_no_duplicate_symmetry():
    spec = PhysicsSpec().symmetry(SymmetryGroup.ROTATION_2D).symmetry(SymmetryGroup.ROTATION_2D)
    d = json.loads(spec.to_json())
    assert d["symmetry_groups"].count("rotation_2d") == 1


def test_conserve_builder():
    spec = PhysicsSpec().conserve("energy").conserve("mass")
    d = json.loads(spec.to_json())
    assert "energy" in d["conserved_quantities"]
    assert "mass"   in d["conserved_quantities"]


def test_boundary_builder():
    spec = PhysicsSpec().boundary("left", BCType.DIRICHLET, value=0.0) \
                        .boundary("right", BCType.NEUMANN, flux=1.0)
    d = json.loads(spec.to_json())
    assert d["boundary_conditions"]["left"]["type"]   == "dirichlet"
    assert d["boundary_conditions"]["right"]["type"]  == "neumann"
    assert abs(d["boundary_conditions"]["right"]["flux"] - 1.0) < 1e-9


def test_grid_builder():
    spec = PhysicsSpec().grid(32, 32, resolution=0.5)
    d = json.loads(spec.to_json())
    assert d["grid_nx"] == 32
    assert d["grid_ny"] == 32
    assert abs(d["spatial_resolution"] - 0.5) < 1e-9


def test_json_roundtrip():
    spec = (PhysicsSpec()
            .pde(PDEType.HEAT, diffusivity=0.02)
            .symmetry(SymmetryGroup.ROTATION_2D)
            .conserve("energy")
            .boundary("wall", BCType.DIRICHLET, value=0.0)
            .grid(8, 8, resolution=1.0))
    s    = spec.to_json()
    spec2 = PhysicsSpec.from_json(s)
    assert spec2.to_json() == s


def test_chaining_returns_self():
    spec = PhysicsSpec()
    result = spec.pde(PDEType.HEAT)
    assert result is spec


def test_repr_contains_pde():
    spec = PhysicsSpec().pde(PDEType.WAVE)
    assert "wave" in repr(spec)
