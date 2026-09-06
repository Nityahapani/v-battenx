import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import json
import pytest
from vbatten_x.physics import PhysicsSpec, SymmetryGroup, PDEType


def test_rotation_2d_in_spec():
    spec = PhysicsSpec().symmetry(SymmetryGroup.ROTATION_2D)
    d    = json.loads(spec.to_json())
    assert "rotation_2d" in d["symmetry_groups"]


def test_multiple_symmetries():
    spec = (PhysicsSpec()
            .symmetry(SymmetryGroup.ROTATION_2D)
            .symmetry(SymmetryGroup.REFLECTION)
            .symmetry(SymmetryGroup.TRANSLATION))
    d = json.loads(spec.to_json())
    assert len(d["symmetry_groups"]) == 3


def test_spec_with_symmetry_and_pde():
    spec = (PhysicsSpec()
            .pde(PDEType.HEAT, diffusivity=0.05)
            .symmetry(SymmetryGroup.ROTATION_2D)
            .conserve("energy"))
    d = json.loads(spec.to_json())
    assert d["pde_type"] == "heat"
    assert "rotation_2d" in d["symmetry_groups"]
    assert "energy" in d["conserved_quantities"]


def test_symmetry_spec_roundtrip():
    spec  = PhysicsSpec().symmetry(SymmetryGroup.ROTATION_2D).symmetry(SymmetryGroup.REFLECTION)
    spec2 = PhysicsSpec.from_json(spec.to_json())
    d2    = json.loads(spec2.to_json())
    assert "rotation_2d" in d2["symmetry_groups"]
    assert "reflection"  in d2["symmetry_groups"]


def test_symmetry_booster_trains():
    from vbatten_x.core import Booster
    rng  = np.random.default_rng(0)
    X    = rng.standard_normal((80, 4)).astype(np.float32)
    y    = X.sum(axis=1).astype(np.float32)
    spec = PhysicsSpec().symmetry(SymmetryGroup.ROTATION_2D)
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).set_physics(spec).train(10)
    assert b.num_stages == 10
    preds = b.predict(X)
    assert preds.shape == (80,)
    assert np.isfinite(preds).all()
