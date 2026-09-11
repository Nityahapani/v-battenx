from __future__ import annotations

import os
import json
import math
from typing import Callable, List, Optional


class EarlyStopping:
    def __init__(self, rounds: int, min_delta: float = 1e-6, save_best: bool = True):
        self.rounds     = rounds
        self.min_delta  = min_delta
        self.save_best  = save_best
        self._best      = float("inf")
        self._wait      = 0
        self._best_iter = 0

    def __call__(self, iteration: int, result) -> bool:
        val = result.value if hasattr(result, "value") else float(result)
        if self._best - val > self.min_delta:
            self._best = val
            self._wait = 0
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
        self.history: List[float] = []

    def __call__(self, iteration: int, residual: float) -> bool:
        self.history.append(float(residual))
        return self.stop_on_converge and residual < self.tol

    @property
    def converged(self) -> bool:
        return bool(self.history) and self.history[-1] < self.tol


class TopologyLogger:
    def __init__(self, log_dir: str = "."):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def __call__(self, iteration: int, mutation_log: dict) -> None:
        path = os.path.join(self.log_dir, f"topology_{iteration:05d}.json")
        with open(path, "w") as f:
            json.dump(mutation_log, f)


class LearningRateScheduler:
    def __init__(self, schedule_fn: Callable[[int], float]):
        self._fn = schedule_fn

    def __call__(self, iteration: int) -> float:
        return self._fn(iteration)

    @staticmethod
    def cosine(initial_lr: float, total_steps: int) -> "LearningRateScheduler":
        return LearningRateScheduler(
            lambda t: initial_lr * 0.5 * (1.0 + math.cos(math.pi * t / total_steps))
        )

    @staticmethod
    def step(initial_lr: float, decay: float, step_every: int) -> "LearningRateScheduler":
        return LearningRateScheduler(
            lambda t: initial_lr * (decay ** (t // step_every))
        )

    @staticmethod
    def exponential(initial_lr: float, decay_rate: float) -> "LearningRateScheduler":
        return LearningRateScheduler(
            lambda t: initial_lr * (decay_rate ** t)
        )

    @staticmethod
    def warmup_cosine(
        warmup_steps: int, initial_lr: float, total_steps: int
    ) -> "LearningRateScheduler":
        def schedule(t: int) -> float:
            if t < warmup_steps:
                return initial_lr * t / max(1, warmup_steps)
            progress = (t - warmup_steps) / max(1, total_steps - warmup_steps)
            return initial_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return LearningRateScheduler(schedule)
