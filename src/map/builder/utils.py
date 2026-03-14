from enum import Enum

from ...theme import (
    CELL_SIZE,
    COLOR_EMPTY, COLOR_OBSTACLE, COLOR_GRID_LINE,
    COLOR_START, COLOR_FINISH, COLOR_INTEREST, COLOR_ENTRANCE, COLOR_LINKED,
)


class ObjectMarker(Enum):
    OBSTACLE = 0
    START = 1
    FINISH = 2
    INTEREST = 3
    ENTRANCE = 4  # Used specifically by the template maps


MARKER_DESIGN = {
    ObjectMarker.START:    ("S", COLOR_START),
    ObjectMarker.FINISH:   ("F", COLOR_FINISH),
    ObjectMarker.INTEREST: ("?", COLOR_INTEREST),
    ObjectMarker.ENTRANCE: ("E", COLOR_ENTRANCE),
}
