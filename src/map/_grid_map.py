from ._base_map import *
import os
import numpy as np


class GridMap(BaseMap):
    def __init__(self, map_type: MapType, width: int, height: int, name: str = "grid_map"):
        super().__init__(MapKind.GRID, map_type, width, height, name)

        self.grid = np.zeros((height, width), dtype=np.int8)
        self._default_sections()

    def _default_sections(self):
        """ Method adds the default map sections needed for the map to be valid """
        # Margin is obstacle
        width, height = self.size
        self.grid[0, :] = 1           # top row
        self.grid[height-1, :] = 1    # bottom row
        self.grid[:, 0] = 1           # left column
        self.grid[:, width-1] = 1     # right column
    
    def update_size(self, width: int, height: int):
        new_grid = np.zeros((height, width), dtype=np.int8)
        min_h = min(height, self.size[1])
        min_w = min(width, self.size[0])
        new_grid[:min_h, :min_w] = self.grid[:min_h, :min_w]
        self.grid = new_grid
        self.size = (width, height)
        self._default_sections()

    def save(self):
        self._save(
            grid=self.grid
        )

    @staticmethod
    def load(file_path: str) -> "GridMap":
        map_dict: dict = BaseMap._load(file_path)
        map_kind = MapKind(map_dict["type"][0])
        map_type = MapType(map_dict["type"][1])

        assert map_kind == MapKind.GRID, f"Invalid map type for GridMap: {map_kind.name}"
        assert "grid" in map_dict.keys(), "Grid map missing \"grid\" information"

        size = tuple(map_dict["size"])
        name = os.path.splitext(os.path.basename(file_path))[0]

        grid_map = GridMap(map_type, size[0], size[1], name)
        grid_map.grid = map_dict["grid"]
        if "interest_points" in map_dict:
            grid_map.interest_points = map_dict["interest_points"]
        if "linked_points" in map_dict:
            grid_map.linked_points = map_dict["linked_points"]

        return grid_map

    def check_validity(self, print_issues: bool = True) -> bool:
        valid = True
        if len(self.start_points) != 0:
            if print_issues: print("Invalid GridMap: Start points not linked")
            valid = False
        if len(self.finish_points) != 0:
            if print_issues: print("Invalid GridMap: Finish points not linked")
            valid = False
        if len(self.linked_points) == 0:
            if print_issues: print("Invalid GridMap: Must contain at least a pair of linked start/finish points.")
            valid = False

        return valid


class GridTemplate(GridMap):
    """
    Grid Template room, used for automatic map generation.

    It uses entrance markers to define possible connections to other templates.
    Each entrance is characterized by:
    - edge: 0 (top), 1 (right), 2 (bottom), 3 (left)
    - position: half-cell index along the edge (actual pos = value * 0.5)

      1 | 2 | 3
    1 |       | 1
    2 |       | 2
    3 |       | 3
      1 | 2 | 3 

    - width: width of the entrance (in cells)
    """
    def __init__(self, width, height, name="grid_template"):
        super().__init__(MapType.TEMPLATE, width, height, name)
        self.entrances = np.zeros((0, 3), dtype=np.int32)  # list of (edge, position, width)

    def _default_sections(self):
        """ Method adds the default map sections needed for the template to be valid """
        # Corners are obstacles
        self.grid[0, 0] = 1
        self.grid[0, -1] = 1
        self.grid[-1, 0] = 1
        self.grid[-1, -1] = 1

    def check_validity(self, print_issues: bool = True) -> bool:
        valid = True
        if self.grid[0, 0] != 1 or self.grid[0, -1] != 1 or self.grid[-1, 0] != 1 or self.grid[-1, -1] != 1:
            if print_issues: print("Invalid GridTemplate: Corners must be obstacles.")
            valid = False
        if np.all(self.grid[0, :]) and np.all(self.grid[-1, :]) and np.all(self.grid[:, 0]) and np.all(self.grid[:, -1]):
            if print_issues: print("Invalid GridTemplate: All edges cannot be completely obstacles.")
            valid = False
        if len(self.interest_points) == 0:
            if print_issues: print("Invalid GridTemplate: Should have at least one interest point defined.")
            valid = False

        return valid

    def get_entrances_on_edge(self, edge) -> list:
        raise NotImplementedError("Method needs implementation.")
    
    def rotate_90(self) -> "GridTemplate":
        raise NotImplementedError("Method needs implementation.")

    def flip_horizontal(self) -> "GridTemplate":
        raise NotImplementedError("Method needs implementation.")
