import heapq
from typing import Dict, List, Optional, Tuple

from .base_policy import BasePolicy


class AStarPolicy(BasePolicy):
    """
    4-connected A* on the agent's full GridMap.

    plan() runs once when the path is empty and returns a list of
    world-coordinate cell-centre waypoints.  The inherited act()
    handles waypoint-following from there.

    Requires full_map_access=True on the AgentConfig.
    """

    def plan(self, agent, dt: float) -> Optional[List[Tuple[float, float]]]:  # noqa: ARG002
        if agent.full_map is None:
            return None

        grid = agent.full_map.grid          # (n_rows, n_cols), 1 = obstacle
        n_rows, n_cols = grid.shape

        # World → grid cell  (world x=col, y=row)
        wx, wy  = float(agent.body.position.x), float(agent.body.position.y)
        gx, gy  = float(agent.goal_pos[0]),      float(agent.goal_pos[1])
        start   = (int(wy), int(wx))             # (row, col)
        goal    = (int(gy), int(gx))

        if start == goal:
            return [(gx + 0.5, gy + 0.5)]

        # Sanity checks
        for r, c in (start, goal):
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                return None
        if grid[goal[0], goal[1]] == 1:
            return None

        # ── A* ────────────────────────────────────────────────────────────────
        def h(r: int, c: int) -> int:
            return abs(r - goal[0]) + abs(c - goal[1])

        # heap entries: (f, g, (row, col))
        open_heap: List[Tuple[int, int, Tuple[int, int]]] = [(h(*start), 0, start)]
        g_cost: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

        while open_heap:
            _, g, current = heapq.heappop(open_heap)

            if g > g_cost.get(current, 2**31):   # stale entry
                continue

            if current == goal:
                path = _reconstruct(came_from, goal)
                # Replace the cell-centre last waypoint with the exact goal position
                # so the agent stops where the goal marker is drawn.
                if path:
                    path[-1] = (float(agent.goal_pos[0]), float(agent.goal_pos[1]))
                return path

            for dr, dc in DIRS:
                nr, nc = current[0] + dr, current[1] + dc
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                if grid[nr, nc] == 1:
                    continue
                new_g = g + 1
                nb = (nr, nc)
                if new_g < g_cost.get(nb, 2**31):
                    g_cost[nb] = new_g
                    came_from[nb] = current
                    heapq.heappush(open_heap, (new_g + h(nr, nc), new_g, nb))

        return None   # no path exists


def _reconstruct(
    came_from: Dict[Tuple[int, int], Tuple[int, int]],
    goal: Tuple[int, int],
) -> List[Tuple[float, float]]:
    """Walk came_from back to the start and return world cell-centre waypoints."""
    path = []
    node = goal
    while node in came_from:
        r, c = node
        path.append((c + 0.5, r + 0.5))   # world (x=col+0.5, y=row+0.5)
        node = came_from[node]
    path.reverse()
    return path
