import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import json
import numpy as np
import tempfile
import pytest
from vbatten_x.core import Booster, MutationEvent
from vbatten_x.sklearn import VBattenXRegressor


def make_data(n=100, d=4, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = X.sum(axis=1).astype(np.float32)
    return X, y


def test_mutation_event_class():
    ev = MutationEvent({"type": "expand_1d_to_2d", "region": 0, "pde_r": 0.05})
    assert ev.type == "expand_1d_to_2d"
    assert ev.region == 0
    assert abs(ev.pde_r - 0.05) < 1e-9


def test_mutation_log_empty_before_save():
    X, y = make_data()
    b    = Booster({"dtdo": "threshold", "learning_rate": 0.1})
    b.set_data(X, y).train(5)
    assert b.get_mutation_log() == []


def test_mutation_log_after_save(tmp_path):
    X, y = make_data()
    b    = Booster({"dtdo": "threshold", "tau_expand": 0.001, "learning_rate": 0.1})
    b.set_data(X, y).train(10)
    path = str(tmp_path / "m.json")
    b.save(path)
    log = b.get_mutation_log()
    assert isinstance(log, list)
    assert len(log) == 10
    for stage_events in log:
        assert isinstance(stage_events, list)


def test_stage_pde_residuals_after_save(tmp_path):
    X, y = make_data()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(8)
    path = str(tmp_path / "m.json")
    b.save(path)
    residuals = b.get_stage_pde_residuals()
    assert len(residuals) == 8
    for r in residuals:
        assert "before" in r and "after" in r


def test_model_json_version_is_3():
    X, y = make_data()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    b.save(path)
    with open(path) as f:
        data = json.load(f)
    assert data["version"] in ("3.0.0", "4.0.0")


def test_dtdo_sklearn_interface():
    X, y = make_data(n=120)
    reg  = VBattenXRegressor(n_estimators=10, dtdo="threshold",
                              tau_expand=0.001, learning_rate=0.1)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (120,)
    assert np.isfinite(preds).all()


def test_complexity_budget_params():
    X, y = make_data(n=100)
    b    = Booster({"dtdo": "router", "max_total_dim": 8, "max_regions": 4,
                    "learning_rate": 0.1})
    b.set_data(X, y).train(10)
    assert b.num_stages == 10
    assert np.isfinite(b.train_loss)


def test_save_load_with_dtdo(tmp_path):
    X, y = make_data()
    b    = Booster({"dtdo": "threshold", "learning_rate": 0.1})
    b.set_data(X, y).train(8)
    p1   = b.predict(X)
    path = str(tmp_path / "dtdo_model.json")
    b.save(path)
    b2   = Booster.load(path)
    p2   = b2.predict(X)
    np.testing.assert_allclose(p1, p2, atol=1e-5)
