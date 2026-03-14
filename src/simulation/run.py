"""
Serializable data classes for MAGNET simulation runs.

Two main objects:
  RunConfig       — what you intend to run (input / recipe)
  SimulationRun   — the complete record of what happened (output / result)

Both are pickle-serializable. SimulationRun is self-contained: it embeds
the full map data so it can be replayed without the original .npz file.
The map file is still referenced by path + sha256 hash so you can verify
whether the map has changed since the run was recorded.
"""
import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Algorithm config
# ---------------------------------------------------------------------------

@dataclass
class AlgorithmConfig:
    """
    Describes how an agent decides what to do.

    type        — fully-qualified class name, e.g. "SimplePolicy", "AStarPlanner",
                  "agents.policies.PPOPolicy"
    params      — arbitrary dict passed to the algorithm constructor.
                  For neural nets you might store {"weights_path": "..."} or
                  embed the weights directly as a numpy array.
    """
    type: str = "NoPolicy"
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent config
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """One agent's full specification."""
    agent_type: str                          # e.g. "SimpleAgent"
    start_pos: Tuple[float, float]           # world coords (x=col, y=row)
    goal_pos: Tuple[float, float]            # world coords
    agent_params: Dict[str, Any] = field(default_factory=dict)  # max_speed, etc.
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    full_map_access: bool = True             # give agent the full GridMap (oracle mode)


# ---------------------------------------------------------------------------
# Run config  (input / recipe)
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Everything needed to (re-)run a simulation from scratch.

    map_path  — absolute path to the .npz map file
    map_hash  — sha256 of the .npz file at the time the config was created;
                used later to detect whether the map has changed
    """
    map_path: str
    map_hash: str
    dt: float
    max_steps: int
    random_seed: Optional[int]
    agents: List[AgentConfig]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # ------------------------------------------------------------------
    @classmethod
    def from_map(
        cls,
        map_path: str,
        agents: List[AgentConfig],
        dt: float = 1.0 / 60.0,
        max_steps: int = 10_000,
        random_seed: Optional[int] = None,
    ) -> "RunConfig":
        """Create a RunConfig and compute the map hash automatically."""
        map_path = os.path.abspath(map_path)
        return cls(
            map_path=map_path,
            map_hash=_hash_file(map_path),
            dt=dt,
            max_steps=max_steps,
            random_seed=random_seed,
            agents=agents,
        )


# ---------------------------------------------------------------------------
# Agent record  (per-agent result)
# ---------------------------------------------------------------------------

@dataclass
class AgentRecord:
    """Recorded outcome for one agent after a run."""
    config: AgentConfig
    trajectory: np.ndarray    # (N, 2) float32, world coords (x, y)
    goal_reached: bool = False
    collided: bool = False
    steps: int = 0


# ---------------------------------------------------------------------------
# SimulationRun  (complete, self-contained result)
# ---------------------------------------------------------------------------

@dataclass
class SimulationRun:
    """
    The full record of a simulation run — self-contained and serializable.

    Embeds the GridMap object so the run can be replayed without the
    original .npz file being present.
    """
    config: RunConfig
    grid_map: Any                           # GridMap — typed as Any to avoid
                                            # importing the full map package here
    agent_records: List[AgentRecord]
    total_steps: int = 0
    duration_s: float = 0.0

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Pickle this run to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "SimulationRun":
        """Load a pickled SimulationRun from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"File does not contain a SimulationRun: {path}")
        return obj

    # ------------------------------------------------------------------
    def verify_map(self) -> bool:
        """
        Check whether the referenced map file still matches the hash stored
        in the config.  Returns True if the map is unchanged (or the file
        doesn't exist — caller decides how to handle that).
        """
        path = self.config.map_path
        if not os.path.isfile(path):
            return False
        return _hash_file(path) == self.config.map_hash

    def load_map_from_file(self):
        """
        Re-load the GridMap from the original .npz file (for re-running).
        Raises FileNotFoundError if the file is missing.
        """
        from ..map._grid_map import GridMap  # late import — keeps this module lightweight
        return GridMap.load(self.config.map_path)

    # ------------------------------------------------------------------
    @property
    def n_agents(self) -> int:
        return len(self.agent_records)

    @property
    def all_goals_reached(self) -> bool:
        return all(r.goal_reached for r in self.agent_records)

    def summary(self) -> str:
        lines = [
            f"SimulationRun — {self.config.created_at}",
            f"  Map      : {self.config.map_path}",
            f"  Map OK   : {self.verify_map()}",
            f"  Agents   : {self.n_agents}",
            f"  Steps    : {self.total_steps}",
            f"  Duration : {self.duration_s:.2f}s",
            f"  All goals: {self.all_goals_reached}",
        ]
        for i, r in enumerate(self.agent_records):
            lines.append(
                f"    Agent {i}: goal={r.goal_reached} "
                f"collided={r.collided} steps={r.steps} "
                f"traj_pts={len(r.trajectory)}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    """Return the sha256 hex digest of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
