"""PID control of the deletion-regularizer weight (Eq. 3).

The objective is ``L = L_CE + alpha_t * L_G`` (Sec. 3.2). To reach a target
deletion rate ``delta``, the weight ``alpha_t`` is set by feedback control on
the error ``e_t = delta - r_t`` between the target and the observed rate:

    alpha_{t+1} = max(0,  k_p * ebar_t  +  k_i * sum_{u<=t} e_u  +  k_d * (e_t - e_{t-omega}) ),
    ebar_t      = rho * ebar_{t-1} + (1 - rho) * e_t.

``t`` counts controller updates (one per optimizer step) and ``omega`` is the
derivative lookback; the D term is zero until ``omega`` past errors exist.
As in MrT5, ``rho = 0.9`` and the non-negative clamp prevents the regularizer
from adding pressure toward retention. Paper settings (Sec. 4.3, App. C):
``k_p = 1``, ``k_i = 1e-5``, ``k_d = 0.5``, ``omega = 10``, ``alpha_0 = 0.1``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PIDConfig:
    k_p: float = 1.0
    k_i: float = 1e-5
    k_d: float = 0.5
    omega: int = 10
    rho: float = 0.9
    alpha_0: float = 0.1


class PIDController:
    """Stateful controller; call :meth:`step` once per optimizer step."""

    def __init__(self, target: float, config: PIDConfig = PIDConfig()) -> None:
        self.target = float(target)
        self.config = config
        self.alpha = float(config.alpha_0)  # alpha_0
        self.ebar = 0.0  # EMA of the error (P term)
        self.integral = 0.0  # sum of errors (I term)
        self.errors: deque[float] = deque(maxlen=config.omega + 1)  # e_{t-omega}, ..., e_t
        self.t = 0

    def step(self, rate: float) -> float:
        """Update with the observed deletion rate ``r_t``; returns ``alpha_{t+1}``."""
        cfg = self.config
        e = self.target - float(rate)  # e_t = delta - r_t
        self.ebar = cfg.rho * self.ebar + (1.0 - cfg.rho) * e  # ebar_t
        self.integral += e  # sum_u e_u
        self.errors.append(e)
        p = cfg.k_p * self.ebar  # P
        i = cfg.k_i * self.integral  # I
        d = cfg.k_d * (self.errors[-1] - self.errors[0]) if len(self.errors) == cfg.omega + 1 else 0.0  # D
        self.alpha = max(0.0, p + i + d)  # [Eq. 3]
        self.t += 1
        return self.alpha

    def state_dict(self) -> dict:
        return {"target": self.target, "alpha": self.alpha, "ebar": self.ebar,
                "integral": self.integral, "errors": list(self.errors), "t": self.t}

    def load_state_dict(self, state: dict) -> None:
        self.target = float(state["target"])
        self.alpha = float(state["alpha"])
        self.ebar = float(state["ebar"])
        self.integral = float(state["integral"])
        self.errors = deque(state["errors"], maxlen=self.config.omega + 1)
        self.t = int(state["t"])


__all__ = ["PIDConfig", "PIDController"]
