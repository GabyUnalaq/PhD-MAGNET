"""
Shared visual constants for the MAGNET project.
Imported by both the map builder and the simulator visualizer.
"""
from PyQt5.QtGui import QColor

# Scale: pixels per world unit (1 world unit = 1 grid cell)
CELL_SIZE = 40

# --- Map / Builder colors ---
COLOR_EMPTY       = QColor("#E0E0E0")
COLOR_OBSTACLE    = QColor("#404040")
COLOR_GRID_LINE   = QColor("#B0B0B0")
COLOR_START       = QColor("#2E7D32")
COLOR_FINISH      = QColor("#C62828")
COLOR_INTEREST    = QColor("#1565C0")
COLOR_ENTRANCE    = QColor("#E7E29C")
COLOR_LINKED      = QColor("#FF8F00")

# --- Simulator / Visualizer colors ---
COLOR_SIM_BG      = COLOR_EMPTY
COLOR_SIM_WALL    = COLOR_GRID_LINE
COLOR_AGENT       = QColor("#0288D1")
COLOR_AGENT_OUT   = QColor("#01579B")
COLOR_HEADING     = QColor("#000000")
COLOR_TRAJECTORY  = COLOR_INTEREST
