from typing import Tuple

import numpy as np

from .base_policy import BasePolicy


class ZoneBoundedPolicy(BasePolicy):
    """
    Wraps any policy and clamps velocity to keep the agent inside a
    rectangular zone defined in world coordinates.

    bounds: (x_min, x_max, y_min, y_max)

    After the inner policy returns a velocity, we check whether the
    resulting position (pos + vel * dt) would leave the zone.  If so,
    the offending component is zeroed out — the agent simply stops at
    the boundary rather than crossing it.
    """

    def __init__(self, inner: BasePolicy, bounds: Tuple[float, float, float, float]):
        super().__init__({})
        self.inner = inner
        self.x_min, self.x_max, self.y_min, self.y_max = bounds

    def act(self, agent, dt: float) -> np.ndarray:
        vel = self.inner.act(agent, dt).copy()

        if agent.body is None:
            return vel

        px = float(agent.body.position.x)
        py = float(agent.body.position.y)

        new_x = px + float(vel[0]) * dt
        new_y = py + float(vel[1]) * dt

        if new_x < self.x_min or new_x > self.x_max:
            vel[0] = 0.0
        if new_y < self.y_min or new_y > self.y_max:
            vel[1] = 0.0

        return vel

    def reset(self) -> None:
        self.inner.reset()
