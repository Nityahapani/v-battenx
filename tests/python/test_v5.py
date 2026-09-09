import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import json
import numpy as np
import tempfile
import pytest
from vbatten_x import (
    Booster, MutationEvent, PhysicalDataset, __version__,
    train, train_with_physics, cv, early_stopping,
    PhysicsSpec, PDEType, SymmetryGroup, BCType,
    EarlyStopping, ModelCheckpoint, PhysicsResidualMonitor,
    TopologyLogger, LearningRateScheduler,
)
from vbatten_x.sklearn import VBattenXRegressor, VBattenXClassifier
from vbatten_x import _libvbatten as _lib


def make_reg(n=150, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = X.sum(axis=1).astype(np.float32)
    return X, y


def make_clf(n=150, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X   = rng.standard_normal((n, d)).astype(np.float32)
    y   = (rng.standard_normal(n) > 0).astype(np.float32)
    return X, y


# ── version & ABI ─────────────────────────────────────────────────────────────

def test_version_is_5():
    assert __version__ == "5.0.0"


def test_abi_version():
    assert _lib.abi_version() == 5


def test_lib_version_string():
    assert _lib.lib_version() == "5.0.0"


def test_model_json_version_is_5(tmp_path):
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(3)
    path = str(tmp_path / "m.json")
    b.save(path)
    with open(path) as f:
        assert json.load(f)["version"] == "5.0.0"


# ── PhysicalDataset ───────────────────────────────────────────────────────────

def test_physical_dataset_from_numpy():
    X, y = make_reg()
    ds   = PhysicalDataset.from_numpy(X, y, feature_names=["a","b","c","d","e"])
    assert ds.shape == (150, 5)
    assert ds.feature_names[0] == "a"


def test_physical_dataset_repr():
    ds = PhysicalDataset(np.zeros((10, 3), dtype=np.float32))
    assert "PhysicalDataset" in repr(ds)


def test_booster_accepts_physical_dataset():
    X, y = make_reg()
    ds   = PhysicalDataset.from_numpy(X, y)
    b    = Booster({"learning_rate": 0.1})
    b.set_data(ds, None).train(5)
    preds = b.predict(ds)
    assert preds.shape == (150,)


def test_train_accepts_physical_dataset():
    X, y = make_reg()
    ds   = PhysicalDataset.from_numpy(X, y)
    b    = train(ds, num_boost_round=5)
    assert b.num_stages == 5


# ── C API completeness ────────────────────────────────────────────────────────

def test_get_metric_train_loss():
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(10)
    metric = b.get_metric("train_loss")
    assert np.isfinite(metric)
    assert metric >= 0.0


def test_get_metric_num_stages():
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(7)
    assert int(b.get_metric("num_stages")) == 7


def test_abi_version_property():
    X, y = make_reg()
    b    = Booster()
    b.set_data(X, y).train(2)
    assert b.abi_version == 5


def test_lib_version_property():
    b = Booster()
    assert b.lib_version == "5.0.0"


# ── callbacks ─────────────────────────────────────────────────────────────────

def test_early_stopping_fires():
    es = EarlyStopping(rounds=3, min_delta=1e-9)
    class FakeResult:
        def __init__(self, v): self.value = v
    for i in range(10):
        fired = es(i, FakeResult(1.0))
        if fired:
            assert es._wait >= 3
            break


def test_model_checkpoint_saves(tmp_path):
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(3)
    ck   = ModelCheckpoint(str(tmp_path / "ck_iter{iteration}.json"), save_period=1)
    ck(1, b)
    assert os.path.exists(str(tmp_path / "ck_iter1.json"))


def test_physics_residual_monitor():
    mon = PhysicsResidualMonitor(tol=0.5, stop_on_converge=True)
    assert not mon(0, 1.0)
    assert not mon(1, 0.6)
    assert mon(2, 0.4)
    assert mon.converged


def test_lr_scheduler_cosine():
    sched = LearningRateScheduler.cosine(0.1, 100)
    assert abs(sched(0)   - 0.1)  < 1e-5
    assert abs(sched(100) - 0.0)  < 1e-4
    assert sched(50) < sched(0)


def test_lr_scheduler_step():
    sched = LearningRateScheduler.step(0.1, 0.5, 10)
    assert abs(sched(0)  - 0.1)   < 1e-6
    assert abs(sched(10) - 0.05)  < 1e-6
    assert abs(sched(20) - 0.025) < 1e-6


def test_topology_logger_creates_files(tmp_path):
    logger = TopologyLogger(str(tmp_path))
    logger(0, {"regions": 1, "edges": 0})
    assert os.path.exists(str(tmp_path / "topology_00000.json"))


# ── field_viz (no-crash) ──────────────────────────────────────────────────────

def test_field_viz_pde_residuals(tmp_path):
    from vbatten_x.field_viz import plot_pde_residuals
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    b.save(str(tmp_path / "m.json"))
    ax = plot_pde_residuals(b)
    assert ax is not None


def test_field_viz_mutation_history(tmp_path):
    from vbatten_x.field_viz import plot_mutation_history
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    b.save(str(tmp_path / "m.json"))
    ax = plot_mutation_history(b)
    assert ax is not None


def test_field_viz_dimension_map(tmp_path):
    from vbatten_x.field_viz import plot_dimension_map
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    b.save(str(tmp_path / "m.json"))
    ax = plot_dimension_map(b, stage=0)
    assert ax is not None


def test_field_viz_field_slice(tmp_path):
    from vbatten_x.field_viz import plot_field_slice
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    b.save(str(tmp_path / "m.json"))
    ax = plot_field_slice(b, stage=0)
    assert ax is not None


# ── sklearn v5 ────────────────────────────────────────────────────────────────

def test_sklearn_regressor_v5():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = make_reg(n=200)
    pipe = Pipeline([("sc", StandardScaler()), ("m", VBattenXRegressor(n_estimators=20))])
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.0


def test_sklearn_check_estimator_regressor():
    # Verify the estimator has all required sklearn attributes and methods
    reg = VBattenXRegressor(n_estimators=5, learning_rate=0.1)
    assert hasattr(reg, 'fit') and callable(reg.fit)
    assert hasattr(reg, 'predict') and callable(reg.predict)
    assert hasattr(reg, 'get_params') and callable(reg.get_params)
    assert hasattr(reg, 'set_params') and callable(reg.set_params)
    params = reg.get_params()
    assert 'n_estimators' in params
    assert 'learning_rate' in params
    reg2 = reg.set_params(n_estimators=10)
    assert reg2.n_estimators == 10


# ── compat ────────────────────────────────────────────────────────────────────

def test_compat_field_params_to_numpy(tmp_path):
    from vbatten_x.compat import field_params_to_numpy
    X, y = make_reg()
    b    = Booster({"learning_rate": 0.1})
    b.set_data(X, y).train(5)
    b.save(str(tmp_path / "m.json"))
    p = field_params_to_numpy(b, stage=0)
    assert isinstance(p, np.ndarray)
    assert len(p) > 0


def test_compat_to_numpy():
    from vbatten_x.compat import to_numpy
    arr = to_numpy([1.0, 2.0, 3.0])
    assert arr.dtype == np.float64


# ── training v5 ───────────────────────────────────────────────────────────────

def test_early_stopping_helper():
    es = early_stopping(rounds=5)
    assert isinstance(es, EarlyStopping)
    assert es.rounds == 5


def test_cv_with_physics():
    X, y = make_reg(n=200)
    spec = PhysicsSpec().pde(PDEType.HEAT, diffusivity=0.1).grid(4, 4)
    result = cv(X, y,
                params={"learning_rate": 0.1, "lambda_pde": 0.01},
                nfold=3, num_boost_round=10)
    assert len(result["test-rmse-mean"]) == 3
    for v in result["test-rmse-mean"]:
        assert np.isfinite(v)
