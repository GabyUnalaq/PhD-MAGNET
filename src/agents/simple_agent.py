from typing import Tuple, List, Optional

from .base_agent import BaseAgent
from .policies.direct_to_goal import DirectToGoalPolicy
from ..map._grid_map import GridMap
from ..sensors import BaseSensor, LidarSensor


class SimpleAgent(BaseAgent):
    """
    Simple agent without facing angle.

    Moves directly toward the next waypoint without rotation constraints.
    Useful for testing path planning algorithms before adding vehicle dynamics.
    """

    def __init__(
        self,
        agent_id: int,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        sensors: Optional[List[BaseSensor]] = None,
        full_map: Optional[GridMap] = None,
        max_speed: float = 1.0
    ):
        """
        Initialize SimpleAgent.

        Args:
            agent_id: Unique identifier
            start_pos: Spawn position in world coordinates
            goal_pos: Goal position in world coordinates
            sensors: List of sensors (defaults to single LidarSensor if None)
            full_map: Optional full map for testing
            max_speed: Maximum movement speed
        """
        # Default sensor
        if sensors is None:
            sensors = [LidarSensor(num_rays=16, max_range=5.0)]

        super().__init__(agent_id, start_pos, goal_pos, sensors, full_map)
        self.MAX_SPEED = max_speed

        # SimpleAgent has no heading dynamics
        self.kinematic.heading = None

        # Default policy: go straight to goal
        self.policy = DirectToGoalPolicy({})

    def _process_observation(self, obs, sensor: BaseSensor):
        """
        Process lidar observation and update local map.

        Args:
            obs: Lidar distances array
            sensor: The LidarSensor that generated the observation
        """
        if not isinstance(sensor, LidarSensor):
            return  # Only handle lidar for now

        # TODO: Convert lidar distances to obstacle cells in local_map
        pass

    def set_path(self, path: List[Tuple[float, float]]):
        """
        Set the planned path for the agent.

        Args:
            path: List of waypoints in world coordinates
        """
        self.navigation.path = path
        self.navigation.current_waypoint_idx = 0
        self.navigation.goal_reached = False
