from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulator.simulator import Simulator


class BaseStopCondition:
    """
    Base class for stop conditions.  Subclass and override __call__, or
    just use a plain callable (lambda / function) — both work.
    """

    def __call__(self, sim: "Simulator") -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AllGoalsReached(BaseStopCondition):
    """Stop when every agent has reached its goal (default)."""

    def __call__(self, sim: "Simulator") -> bool:
        return bool(sim.agents) and all(
            a.navigation.goal_reached for a in sim.agents
        )


class AnyGoalReached(BaseStopCondition):
    """Stop as soon as any single agent reaches its goal."""

    def __call__(self, sim: "Simulator") -> bool:
        return any(a.navigation.goal_reached for a in sim.agents)


class MaxSteps(BaseStopCondition):
    """Stop after a fixed number of simulation steps."""

    def __init__(self, max_steps: int):
        self.max_steps = max_steps

    def __call__(self, sim: "Simulator") -> bool:
        return sim._step_count >= self.max_steps

    def __repr__(self) -> str:
        return f"MaxSteps({self.max_steps})"
