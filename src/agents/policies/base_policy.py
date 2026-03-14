from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..base_agent import BaseAgent


class BasePolicy:
    """
    Decision-making component attached to an agent.

    Two override points — implement whichever suits your algorithm:

      plan(agent, dt)  →  List[waypoints] | None
          Path-planning algorithms (A*, RRT, DWA, …) override this.
          Return world-coordinate (x, y) waypoints; the default act()
          will follow them using waypoint-following control.
          Return None to keep the current path unchanged.

      act(agent, dt)  →  ndarray(vx, vy)
          Direct-control algorithms (RL, APF reactive, …) override this.
          Return the desired velocity vector; no waypoints needed.
          The default implementation calls plan() when the path is empty
          and then applies waypoint-following control.
    """

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def plan(self, agent: "BaseAgent", dt: float) -> Optional[List[Tuple[float, float]]]:
        """
        Compute a path from agent's current position to its goal.

        Return a list of (x, y) world-coordinate waypoints, or None to leave
        agent.navigation.path unchanged.  Called by the default act() when the
        current path is exhausted.
        """
        return None

    def act(self, agent: "BaseAgent", dt: float) -> np.ndarray:
        """
        Return desired velocity (vx, vy), shape (2,), dtype float32.

        Default: call plan() when the path runs out, then follow waypoints.
        Override entirely for direct-control policies (RL, APF, …).
        """
        nav = agent.navigation
        if nav.goal_reached:
            return np.zeros(2, dtype=np.float32)

        if not nav.path or nav.current_waypoint_idx >= len(nav.path):
            new_path = self.plan(agent, dt)
            if new_path:
                nav.path = list(new_path)
                nav.current_waypoint_idx = 0

        return self._follow_waypoints(agent, dt)

    def reset(self) -> None:
        """Called on episode reset. Override if the policy has internal state."""
        pass

    def _follow_waypoints(self, agent: "BaseAgent", dt: float) -> np.ndarray:
        """Proportional waypoint-following controller (shared by all planners)."""
        nav = agent.navigation

        if not nav.path or nav.current_waypoint_idx >= len(nav.path):
            return np.zeros(2, dtype=np.float32)

        wp_world = np.array(nav.path[nav.current_waypoint_idx], dtype=np.float32)
        wp_local = agent.world_to_agent_local(wp_world)
        to_wp = wp_local - agent.kinematic.position
        dist = float(np.linalg.norm(to_wp))

        if dist < 0.1:
            nav.current_waypoint_idx += 1
            if nav.current_waypoint_idx >= len(nav.path):
                agent.on_goal_reached()
                return np.zeros(2, dtype=np.float32)
            wp_world = np.array(nav.path[nav.current_waypoint_idx], dtype=np.float32)
            wp_local = agent.world_to_agent_local(wp_world)
            to_wp = wp_local - agent.kinematic.position
            dist = float(np.linalg.norm(to_wp))

        if dist > 0:
            return (to_wp / dist) * min(agent.MAX_SPEED, dist / dt)
        return np.zeros(2, dtype=np.float32)
