import time
from typing import Callable, List, Optional, Union

import numpy as np
import pymunk

from ..agents import BaseAgent, SimpleAgent
from ..agents.policies import create as create_policy
from ..map._grid_map import GridMap
from ..simulation.run import AgentRecord, RunConfig, SimulationRun
from ..simulation.stop_conditions import AllGoalsReached, BaseStopCondition

StopConditionT = Union[BaseStopCondition, Callable]


class Simulator:
    def __init__(self, headless: bool = True):
        self.space = pymunk.Space()
        self.headless = headless
        self.renderer = None
        self.agents: List[BaseAgent] = []
        self.stop_condition: StopConditionT = AllGoalsReached()

        self._config: Optional[RunConfig] = None
        self._grid_map: Optional[GridMap] = None
        self._step_count: int = 0
        self._start_time: float = 0.0

    @property
    def is_done(self) -> bool:
        """True when the current stop condition is satisfied."""
        return self.stop_condition(self)

    # Setup ---------------------------------------------------------------------

    def setup(self, config: RunConfig, grid_map: Optional[GridMap] = None):
        """
        Configure the simulator from a RunConfig.

        If grid_map is provided it is used directly (avoids a redundant file
        read when the caller already has the map loaded).  Otherwise the map
        is loaded from config.map_path.

        Returns self for chaining.
        """
        if grid_map is None:
            grid_map = GridMap.load(config.map_path)
        self.load_map(grid_map)
        self.spawn_agents_from_config(config, grid_map)
        self._config = config
        self._step_count = 0
        self._start_time = time.monotonic()
        return self

    def load_map(self, grid_map: GridMap):
        """
        Populate the Pymunk space with static shapes from a GridMap.
        Each obstacle cell (grid value == 1) becomes a static polygon.

        Coordinate convention: world x = col, world y = row.
        Matches the builder's screen layout so the visualizer renders
        identically to the map builder.
        """
        self.clear()
        self._grid_map = grid_map
        static_body = self.space.static_body
        w, h = grid_map.size  # w = num cols (x axis), h = num rows (y axis)

        for row in range(h):
            for col in range(w):
                if grid_map.grid[row, col] == 1:
                    x, y = float(col), float(row)
                    verts = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
                    shape = pymunk.Poly(static_body, verts)
                    self.space.add(shape)

    def spawn_agents(self, grid_map: GridMap):
        """
        Create one SimpleAgent per linked start/finish pair.

        linked_points row: [start_gx, start_gy, finish_gx, finish_gy]
        where gx = row index, gy = col index in map space.
        World position: x = gy (col), y = gx (row).
        Agents receive the full map so sensor observations are skipped.
        """
        for i, lp in enumerate(grid_map.linked_points):
            start_world = (float(lp[1]), float(lp[0]))  # (x=col, y=row)
            goal_world  = (float(lp[3]), float(lp[2]))
            agent = SimpleAgent(
                agent_id=i,
                start_pos=start_world,
                goal_pos=goal_world,
                full_map=grid_map,
            )
            self._add_agent_body(agent, start_world)

    def spawn_agents_from_config(self, config: RunConfig, grid_map: GridMap):
        """
        Create agents as specified in a RunConfig.
        Currently supports SimpleAgent; extend as new agent types are added.
        """
        for agent_cfg in config.agents:
            if agent_cfg.agent_type == "SimpleAgent":
                agent = SimpleAgent(
                    agent_id=len(self.agents),
                    start_pos=agent_cfg.start_pos,
                    goal_pos=agent_cfg.goal_pos,
                    full_map=grid_map if agent_cfg.full_map_access else None,
                    **{k: v for k, v in agent_cfg.agent_params.items()
                       if k in ("max_speed",)},
                )
            else:
                raise ValueError(f"Unknown agent type: {agent_cfg.agent_type!r}")
            agent.policy = create_policy(agent_cfg.algorithm)
            self._add_agent_body(agent, agent_cfg.start_pos)

    def clear(self):
        """Remove all shapes, non-static bodies, and agents from the space."""
        for shape in list(self.space.shapes):
            self.space.remove(shape)
        for body in list(self.space.bodies):
            # space.bodies never includes static_body, safe to remove all
            self.space.remove(body)
        self.agents.clear()
        self._grid_map = None
        self._config = None

    # Loop ----------------------------------------------------------------------

    def step(self, dt: float):
        for agent in self.agents:
            agent.step(dt)
        self.space.step(dt)
        self._step_count += 1

        if not self.headless:
            self.renderer.draw()

    def run_headless(self) -> "SimulationRun":
        """
        Run the full simulation to completion without any UI and return the result.

        Stops early when all agents have reached their goals or the configured
        max_steps limit is hit.  Call setup() first.
        """
        if self._config is None:
            raise RuntimeError("Call setup() before run_headless().")
        dt = self._config.dt
        max_steps = self._config.max_steps
        while self._step_count < max_steps:
            self.step(dt)
            if self.is_done:
                break
        return self.finalize()

    # Result collection ---------------------------------------------------------

    def finalize(self) -> SimulationRun:
        """
        Collect trajectories and stats from all agents and return a complete,
        pickle-ready SimulationRun.  Does NOT save to disk — call run.save(path).
        """
        if self._config is None or self._grid_map is None:
            raise RuntimeError("Call setup() or load_map() before finalize().")

        duration = time.monotonic() - self._start_time
        records = []

        for agent, agent_cfg in zip(self.agents, self._config.agents):
            traj = np.array(list(agent.trajectory), dtype=np.float32)
            records.append(AgentRecord(
                config=agent_cfg,
                trajectory=traj,
                goal_reached=agent.navigation.goal_reached,
                collided=agent.kinematic.collided,
                steps=len(agent.trajectory),
            ))

        return SimulationRun(
            config=self._config,
            grid_map=self._grid_map,
            agent_records=records,
            total_steps=self._step_count,
            duration_s=duration,
        )

    # Internal helpers ----------------------------------------------------------

    def _add_agent_body(self, agent: BaseAgent, world_pos: tuple):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = world_pos
        shape = pymunk.Circle(body, 0.3)
        self.space.add(body, shape)
        agent.body = body
        agent.shape = shape
        agent.space = self.space
        self.agents.append(agent)
