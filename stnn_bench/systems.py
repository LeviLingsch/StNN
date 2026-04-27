from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class GeneratedTrajectories:
    t_eval: np.ndarray
    trajectories: np.ndarray


class DynamicalSystem(ABC):
    """Abstract interface for continuous-time systems solved with solve_ivp."""

    name: str
    state_dim: int
    state_labels: Sequence[str]
    observable_indices: Sequence[int]

    def __init__(self, *, rtol: float = 1e-8, atol: float = 1e-8) -> None:
        self.rtol = rtol
        self.atol = atol

    @abstractmethod
    def rhs(self, t: float, y: np.ndarray) -> List[float]:
        pass

    @abstractmethod
    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        pass

    def generate_trajectories(
        self,
        *,
        num_trajectories: int,
        t_span: Tuple[float, float],
        dt: float,
        seed: int,
    ) -> GeneratedTrajectories:
        rng = np.random.default_rng(seed)
        t_eval = np.arange(t_span[0], t_span[1] + 1e-12, dt, dtype=np.float64)

        trajectories = np.zeros(
            (num_trajectories, t_eval.shape[0], self.state_dim), dtype=np.float32
        )

        for idx in range(num_trajectories):
            y0 = self.sample_initial_state(rng)
            solution = solve_ivp(
                self.rhs,
                t_span,
                y0,
                t_eval=t_eval,
                rtol=self.rtol,
                atol=self.atol,
            )
            if not solution.success:
                raise RuntimeError(
                    f"Trajectory generation failed for {self.name}: {solution.message}"
                )
            trajectories[idx] = solution.y.T.astype(np.float32)

        return GeneratedTrajectories(t_eval=t_eval.astype(np.float32), trajectories=trajectories)


class DampedPendulumSystem(DynamicalSystem):
    name = "damped_pendulum"
    state_dim = 2
    state_labels = ("theta", "omega")
    observable_indices = (0, 1)

    def __init__(self, damping: float = 0.15, gravity_over_length: float = 1.0) -> None:
        super().__init__()
        self.damping = damping
        self.gravity_over_length = gravity_over_length

    def rhs(self, t: float, y: np.ndarray) -> List[float]:
        theta, omega = y
        return [omega, -self.damping * omega - self.gravity_over_length * np.sin(theta)]

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        theta = rng.uniform(-np.pi, np.pi)
        omega = rng.uniform(-1.5, 1.5)
        return np.array([theta, omega], dtype=np.float64)


class DuffingSystem(DynamicalSystem):
    name = "duffing"
    state_dim = 2
    state_labels = ("x", "v")
    observable_indices = (0, 1)

    def __init__(self, alpha: float = -1.0, beta: float = 1.0, delta: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta

    def rhs(self, t: float, y: np.ndarray) -> List[float]:
        x, v = y
        dx = v
        dv = -self.delta * v - self.alpha * x - self.beta * (x ** 3)
        return [dx, dv]

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(-2.0, 2.0)
        v = rng.uniform(-1.5, 1.5)
        return np.array([x, v], dtype=np.float64)


class ForcedDuffingSystem(DynamicalSystem):
    """Autonomous form of forced Duffing with a phase state."""

    name = "forced_duffing"
    state_dim = 3
    state_labels = ("x", "v", "phi")
    observable_indices = (0, 1)

    def __init__(
        self,
        alpha: float = -1.0,
        beta: float = 1.0,
        delta: float = 0.2,
        gamma: float = 0.3,
        omega: float = 1.2,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.omega = omega

    def rhs(self, t: float, y: np.ndarray) -> List[float]:
        x, v, phi = y
        dx = v
        dv = -self.delta * v - self.alpha * x - self.beta * (x ** 3) + self.gamma * np.cos(phi)
        dphi = self.omega
        return [dx, dv, dphi]

    def sample_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(-2.0, 2.0)
        v = rng.uniform(-1.5, 1.5)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        return np.array([x, v, phi], dtype=np.float64)


def build_system(system_cfg: Dict[str, object]) -> DynamicalSystem:
    system_name = str(system_cfg.get("name", "")).strip().lower()
    params = dict(system_cfg.get("params", {}))

    if system_name in {"pendulum", "damped_pendulum"}:
        return DampedPendulumSystem(
            damping=float(params.get("damping", 0.15)),
            gravity_over_length=float(params.get("gravity_over_length", 1.0)),
        )

    if system_name in {"duffing", "unforced_duffing"}:
        return DuffingSystem(
            alpha=float(params.get("alpha", -1.0)),
            beta=float(params.get("beta", 1.0)),
            delta=float(params.get("delta", 0.2)),
        )

    if system_name in {"forced_duffing", "duffing_forced"}:
        return ForcedDuffingSystem(
            alpha=float(params.get("alpha", -1.0)),
            beta=float(params.get("beta", 1.0)),
            delta=float(params.get("delta", 0.2)),
            gamma=float(params.get("gamma", 0.3)),
            omega=float(params.get("omega", 1.2)),
        )

    raise ValueError(f"Unknown system name: {system_name}")
