import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, roc_auc_score

from vbatten_x.sklearn import VBattenXRegressor, VBattenXClassifier


def test_regressor_fit_predict():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=0)
    X = X.astype(np.float32)
    reg = VBattenXRegressor(n_estimators=30, learning_rate=0.1)
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (200,)


def test_regressor_score_positive():
    X, y = make_regression(n_samples=300, n_features=6, noise=0.05, random_state=1)
    X = X.astype(np.float32)
    reg = VBattenXRegressor(n_estimators=50, learning_rate=0.1)
    reg.fit(X, y)
    assert reg.score(X, y) > 0.0


def test_classifier_fit_predict_proba():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    clf = VBattenXClassifier(n_estimators=30, learning_rate=0.05)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_classifier_auc_above_random():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = VBattenXClassifier(n_estimators=50, learning_rate=0.05)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
    assert auc > 0.7


def test_pipeline_compatible():
    X, y = make_regression(n_samples=200, n_features=5, random_state=2)
    X = X.astype(np.float32)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  VBattenXRegressor(n_estimators=20)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (200,)


def test_get_set_params():
    reg = VBattenXRegressor(n_estimators=50, learning_rate=0.05)
    params = reg.get_params()
    assert params["n_estimators"] == 50
    reg.set_params(n_estimators=100)
    assert reg.n_estimators == 100
