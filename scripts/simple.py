import sys
import os
from PyQt5.QtWidgets import QApplication

from src.agents.policies import AlgorithmConfig
from src.viz import VisualizerWindow
from src.map._grid_map import GridMap
from src.map import MAPS_FILES_DIR, EXT
from src.simulator import Simulator
from src.simulation.run import AgentConfig, RunConfig

# ── Test map ──────────────────────────────────────────────────────────────────
TEST_MAP = os.path.join(MAPS_FILES_DIR, f"test_ui.{EXT}")


def main():
    app = QApplication(sys.argv)
    window = VisualizerWindow()
    window.show()

    # Load map
    grid_map = GridMap.load(TEST_MAP)
    w, h = grid_map.size
    n_pairs = len(grid_map.linked_points)
    window.log(f"Map: {grid_map.name}  ({w}x{h})  |  {n_pairs} linked pair(s)")

    if n_pairs == 0:
        window.log("No linked start/finish pairs — add some in the map builder.")
        sys.exit(app.exec_())

    # Build RunConfig from the map's linked pairs
    # linked_points: [gx=row, gy=col, gx2, gy2]  →  world: x=col, y=row
    agents = [
        AgentConfig(
            agent_type="SimpleAgent",
            start_pos=(float(lp[1]), float(lp[0])),
            goal_pos =(float(lp[3]), float(lp[2])),
            algorithm=AlgorithmConfig("DirectToGoalPolicy")
        )
        for lp in grid_map.linked_points
    ]
    config = RunConfig.from_map(TEST_MAP, agents)
    window.log(f"RunConfig built  |  map hash: {config.map_hash[:12]}…")

    # Set up simulator
    sim = Simulator(headless=True)
    sim.setup(config, grid_map=grid_map)
    window.log(f"Simulator ready  |  {len(sim.agents)} agent(s) spawned")

    # Attach to visualizer and start
    window.set_simulator(sim)
    window.simView.fit_view(w, h)
    window.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
