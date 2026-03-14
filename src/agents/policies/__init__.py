from typing import Any, Dict

from .base_policy import BasePolicy
from .no_policy import NoPolicy
from .direct_to_goal import DirectToGoalPolicy
from .astar_policy import AStarPolicy

from ...simulation.run import AlgorithmConfig

REGISTRY: Dict[str, type] = {
    "NoPolicy": NoPolicy,
    "DirectToGoalPolicy": DirectToGoalPolicy,
    "AStarPolicy": AStarPolicy,
}


def create(config: AlgorithmConfig) -> BasePolicy:
    """Instantiate a policy from an AlgorithmConfig."""
    cls = REGISTRY.get(config.type)
    if cls is None:
        raise KeyError(
            f"Unknown policy {config.type!r}. Available: {sorted(REGISTRY)}"
        )
    return cls(config.params)


__all__ = [
    "BasePolicy",
    "NoPolicy",
    "DirectToGoalPolicy",
    "AStarPolicy",
    "REGISTRY",
    "create",
]
