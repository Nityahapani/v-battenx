from __future__ import annotations

import os
from typing import Optional


class EarlyStopping:
    def __init__(self, rounds: int, min_delta: float = 1e-6, save_best: bool = True):
        self.rounds     = rounds
        self.min_delta  = min_delta
        self.save_best  = save_best
        self._best      = float("inf")
        self._wait      = 0
        self._best_iter = 0

    def __call__(self, iteration: int, result) -> bool:
        val = result.value if hasattr(result, "value") else result
        if self._best - val > self.min_delta:
            self._best      = val
            self._wait      = 0
            self._best_iter = iteration
        else:
            self._wait += 1
        return self._wait >= self.rounds

    @property
    def best_iteration(self) -> int:
        return self._best_iter


class ModelCheckpoint:
    def __init__(self, path: str, save_period: int = 10):
        self.path        = path
        self.save_period = save_period

    def __call__(self, iteration: int, booster) -> None:
        if (iteration + 1) % self.save_period == 0:
            p = self.path.format(iteration=iteration)
            os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
            booster.save(p)


class PhysicsResidualMonitor:
    def __init__(self, tol: float = 1e-3, stop_on_converge: bool = False):
        self.tol              = tol
        self.stop_on_converge = stop_on_converge
        self.history: list    = []

    def __call__(self, iteration: int, residual: float) -> bool:
        self.history.append(residual)
        if self.stop_on_converge and residual < self.tol:
            return True
        return False


class TopologyLogger:
    def __init__(self, log_dir: str = "."):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def __call__(self, iteration: int, mutation_log: dict) -> None:
        import json
        path = os.path.join(self.log_dir, f"topology_{iteration:05d}.json")
        with open(path, "w") as f:
            json.dump(mutation_log, f)
