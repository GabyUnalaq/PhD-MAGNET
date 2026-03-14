import numpy as np
import pymunk
from typing import TYPE_CHECKING, Tuple, List, Optional
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod

from ..map._grid_map import GridMap
from ..map._base_map import MapType
from ..sensors import BaseSensor

if TYPE_CHECKING:
    from .policies import BasePolicy

MAP_CHUNK_LENGTH = 10
MAX_TRAJECTORY_LENGTH = 10000


@dataclass
class KinematicState:
    """Physical state of the agent."""
    position: np.ndarray  # (x, y) in agent-local coords (0,0 = spawn point)
    velocity: np.ndarray  # (vx, vy)
    heading: Optional[float] = 0.0  # facing angle in radians (None for SimpleAgent)
    alive: bool = True
    collided: bool = False


@dataclass
class NavigationState:
    """Navigation and planning state."""
    path: List[Tuple[float, float]] = field(default_factory=list)  # planned waypoints
    current_waypoint_idx: int = 0
    goal_reached: bool = False


class BaseAgent(ABC):
    """
    Base class for all agents.

    Manages:
    - Local chunked map with automatic expansion
    - Agent state (kinematic + navigation)
    - Sensors for observation
    - Trajectory tracking
    """

    MAX_SPEED: float = 1.0

    def __init__(
        self,
        agent_id: int,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        sensors: List[BaseSensor],
        full_map: Optional[GridMap] = None
    ):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier
            start_pos: Spawn position in world coordinates
            goal_pos: Goal position in world coordinates
            sensors: List of sensors for observation
            full_map: Optional full map for testing (if None, agent builds map from observations)
        """
        self.id = agent_id
        self.sensors = sensors

        # State
        self.kinematic = KinematicState(
            position=np.array([0.0, 0.0], dtype=np.float32),  # Agent-local: start = (0, 0)
            velocity=np.array([0.0, 0.0], dtype=np.float32),
            heading=0.0
        )
        self.navigation = NavigationState()
        self.goal_pos = np.array(goal_pos, dtype=np.float32)

        # Trajectory tracking — stored in world coordinates
        self._start_pos = np.array(start_pos, dtype=np.float32)
        self.trajectory = deque(maxlen=MAX_TRAJECTORY_LENGTH)
        self.trajectory.append(self._start_pos.copy())

        # Local map with chunking
        self.local_map = GridMap(
            MapType.MAP,
            MAP_CHUNK_LENGTH,
            MAP_CHUNK_LENGTH,
            f"local_map_{agent_id}"
        )
        self.map_offset = np.array(
            [start_pos[0] - MAP_CHUNK_LENGTH // 2, start_pos[1] - MAP_CHUNK_LENGTH // 2],
            dtype=np.float32
        )  # World coords of local_map[0, 0]

        # Full map knowledge (for testing without sensors)
        self.full_map = full_map

        # Pymunk body (set by simulator)
        self.body: Optional[pymunk.Body] = None
        self.shape: Optional[pymunk.Shape] = None

        # Policy (set by simulator from AgentConfig.algorithm)
        self.policy: Optional["BasePolicy"] = None

    def step(self, dt: float):
        """
        Advance agent by one time step.

        Args:
            dt: Time step in seconds
        """
        # 1. Sense: update local_map from sensors
        self._update_observations()

        # 2. Act: compute desired velocity (from external planner or RL policy)
        desired_vel = self._compute_action(dt)

        # 3. Apply to Pymunk body
        if self.body is not None:
            self.body.velocity = pymunk.Vec2d(desired_vel[0], desired_vel[1])

        # 4. Update kinematic state from body
        self._update_state_from_body()

        # 5. Record trajectory in world coords (freeze once goal is reached)
        if not self.navigation.goal_reached and self.body is not None:
            self.trajectory.append(
                np.array([self.body.position.x, self.body.position.y], dtype=np.float32)
            )

        # 6. Check for map expansion
        self._maybe_expand_map()

    def _update_observations(self):
        """Update local map from sensor observations."""
        if self.full_map is not None:
            # No need for observations if full map is available
            return

        for sensor in self.sensors:
            if self.body is not None:
                obs = sensor.observe(self.space, self.body)
                self._process_observation(obs, sensor)

    @abstractmethod
    def _process_observation(self, obs, sensor: BaseSensor):
        """
        Process sensor observation and update local map.

        Args:
            obs: Observation data from sensor
            sensor: The sensor that generated the observation
        """
        pass

    def _compute_action(self, dt: float) -> np.ndarray:
        """Delegate to the attached policy, or stand still if none is set."""
        if self.policy is not None:
            return self.policy.act(self, dt)
        return np.zeros(2, dtype=np.float32)

    def _update_state_from_body(self):
        """Sync kinematic state from Pymunk body."""
        if self.body is None:
            return

        # Update position (in agent-local coords)
        world_pos = np.array([self.body.position.x, self.body.position.y], dtype=np.float32)
        self.kinematic.position = self.world_to_agent_local(world_pos)

        # Update velocity
        self.kinematic.velocity = np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32)

        # Update heading (only for agents that use rotation)
        if self.kinematic.heading is not None:
            self.kinematic.heading = self.body.angle

    def _maybe_expand_map(self):
        """Check if map needs expansion based on agent's current position."""
        if self.body is not None:
            world_pos = np.array([self.body.position.x, self.body.position.y], dtype=np.float32)
            self._expand_map(world_pos)

    def _expand_map(self, point: np.ndarray):
        """
        Expand local map if point is outside current bounds.

        Handles corner cases by expanding in multiple directions at once.

        Args:
            point: Position in world coordinates to check
        """
        # Convert to map-relative coordinates
        relative = point - self.map_offset

        # Check which edges need expansion
        expand_left = relative[0] < 0
        expand_right = relative[0] >= self.local_map.size[0]
        expand_top = relative[1] < 0
        expand_bottom = relative[1] >= self.local_map.size[1]

        if not (expand_left or expand_right or expand_top or expand_bottom):
            return  # Point is within current map

        # Calculate how many chunks needed in each direction
        chunks_left = int(np.ceil(-relative[0] / MAP_CHUNK_LENGTH)) if expand_left else 0
        chunks_right = int(np.ceil((relative[0] - self.local_map.size[0] + 1) / MAP_CHUNK_LENGTH)) if expand_right else 0
        chunks_top = int(np.ceil(-relative[1] / MAP_CHUNK_LENGTH)) if expand_top else 0
        chunks_bottom = int(np.ceil((relative[1] - self.local_map.size[1] + 1) / MAP_CHUNK_LENGTH)) if expand_bottom else 0

        # Calculate new map dimensions
        new_width = self.local_map.size[0] + (chunks_left + chunks_right) * MAP_CHUNK_LENGTH
        new_height = self.local_map.size[1] + (chunks_top + chunks_bottom) * MAP_CHUNK_LENGTH

        # Create new expanded grid
        new_grid = np.zeros((new_height, new_width), dtype=np.int8)

        # Calculate where to copy old data in the new grid
        old_start_row = chunks_top * MAP_CHUNK_LENGTH
        old_start_col = chunks_left * MAP_CHUNK_LENGTH
        old_h, old_w = self.local_map.grid.shape

        # Copy old grid data to new position
        new_grid[old_start_row:old_start_row+old_h, old_start_col:old_start_col+old_w] = self.local_map.grid

        # Update map
        self.local_map.grid = new_grid
        self.local_map.size = (new_width, new_height)

        # Update offset (top-left corner shifts if we added chunks to left or top)
        self.map_offset[0] -= chunks_left * MAP_CHUNK_LENGTH
        self.map_offset[1] -= chunks_top * MAP_CHUNK_LENGTH

    # Coordinate transforms
    def world_to_agent_local(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to agent-local (spawn = 0,0)."""
        spawn_world = self.map_offset + np.array([MAP_CHUNK_LENGTH // 2, MAP_CHUNK_LENGTH // 2])
        return world_pos - spawn_world

    def agent_local_to_world(self, agent_pos: np.ndarray) -> np.ndarray:
        """Convert agent-local coordinates to world coordinates."""
        spawn_world = self.map_offset + np.array([MAP_CHUNK_LENGTH // 2, MAP_CHUNK_LENGTH // 2])
        return agent_pos + spawn_world

    def world_to_chunk(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to local_map indices."""
        relative = world_pos - self.map_offset
        row = int(relative[1])
        col = int(relative[0])
        return (row, col)

    def chunk_to_world(self, row: int, col: int) -> np.ndarray:
        """Convert local_map indices to world coordinates (cell center)."""
        return self.map_offset + np.array([col + 0.5, row + 0.5], dtype=np.float32)

    def on_collision(self, _other):
        """
        Called when agent collides with an obstacle.

        Override in subclass for custom collision behavior.

        Args:
            other: The other Pymunk shape involved in collision
        """
        self.kinematic.collided = True
        self.kinematic.alive = False

    def on_goal_reached(self):
        """
        Called when agent reaches the goal.

        Override in subclass for custom goal behavior.
        """
        self.navigation.goal_reached = True
