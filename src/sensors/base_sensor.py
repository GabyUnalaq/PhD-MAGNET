from abc import ABC, abstractmethod
from typing import Any
import pymunk


class BaseSensor(ABC):
    """
    Abstract base class for all sensors.

    Sensors observe the environment and return observations that agents
    can use to update their local maps or make decisions.
    """

    @abstractmethod
    def observe(self, space: pymunk.Space, agent_body: pymunk.Body) -> Any:
        """
        Observe the environment and return sensor data.

        Args:
            space: The Pymunk physics space containing all objects
            agent_body: The agent's Pymunk body (for position/orientation)

        Returns:
            Sensor-specific observation data
        """
        pass
