import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import pytest
from vbatten_x.core import Booster


def test_booster_default_objective():
    b = Booster()
    assert b.num_stages == 0
    assert b.train_loss == 0.0


def test_classification_objective():
    import numpy as np
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 4)).astype(np.float32)
    y = (rng.standard_normal(100) > 0).astype(np.float32)
    b = Booster({"objective": "classification", "learning_rate": 0.1})
    b.set_data(X, y).train(10)
    assert b.num_stages == 10
    preds = b.predict(X)
    assert preds.shape == (100,)


def test_params_passed_through():
    b = Booster({"learning_rate": 0.05, "reg_lambda": 2.0})
    assert b._params["learning_rate"] == 0.05
    assert b._params["reg_lambda"] == 2.0
