from typing import List, Tuple

from .base_policy import BasePolicy


class DirectToGoalPolicy(BasePolicy):
    """
    Move directly toward the goal in a straight line and stop.

    plan() returns a single-waypoint path [goal_pos]; the base act()
    handles waypoint-following velocity from there.
    """

    def plan(self, agent, dt: float) -> List[Tuple[float, float]]:
        return [tuple(agent.goal_pos)]
