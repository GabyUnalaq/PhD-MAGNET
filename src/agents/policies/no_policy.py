from typing import Any, Dict

import numpy as np

from .base_policy import BasePolicy


class NoPolicy(BasePolicy):
    """Agent stands still — useful as a placeholder or baseline."""

    def act(self, agent, dt: float) -> np.ndarray:
        return np.zeros(2, dtype=np.float32)
