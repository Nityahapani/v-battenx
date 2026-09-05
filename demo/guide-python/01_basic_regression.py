import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python-package"))

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

from vbatten_x.sklearn import VBattenXRegressor

rng = np.random.default_rng(42)
X, y, coef = make_regression(n_samples=2000, n_features=20, n_informative=15,
                              noise=0.5, coef=True, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
X_te   = scaler.transform(X_te).astype(np.float32)

model = VBattenXRegressor(n_estimators=100, learning_rate=0.1, reg_lambda=1.0)
model.fit(X_tr, y_tr)

preds    = model.predict(X_te)
rmse     = root_mean_squared_error(y_te, preds)
baseline = root_mean_squared_error(y_te, np.full_like(y_te, y_tr.mean()))

print(f"V-BATTEN-X  RMSE : {rmse:.4f}")
print(f"Mean baseline RMSE: {baseline:.4f}")
print(f"Improvement:        {(baseline - rmse) / baseline * 100:.1f}%")
print(f"Stages trained:     {model.booster_.num_stages}")
