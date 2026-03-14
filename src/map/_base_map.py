from enum import Enum
import os
import numpy as np

from . import MAPS_FILES_DIR


__all__ = ["MapType", "MapKind", "BaseMap"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class MapKind(Enum):
    """
    Enumeration for the different kinds of maps.
    """
    GRID = 0
    CONTINUOUS = 1

    @staticmethod
    def from_string(value: str) -> "MapKind":
        value = value.lower().replace(" ", "_").strip()
        if value == "grid":
            return MapKind.GRID
        elif value == "continuous":
            return MapKind.CONTINUOUS
        else:
            raise ValueError(f"Unknown map kind string: {value}")
    


class MapType(Enum):
    """
    Enumeration for the different types of maps.
    They can be either maps or templates.
    """
    MAP = 0
    TEMPLATE = 1

    @staticmethod
    def from_string(value: str) -> "MapType":
        value = value.lower().replace(" ", "_").strip()
        if value == "map":
            return MapType.MAP
        elif value == "template":
            return MapType.TEMPLATE
        else:
            raise ValueError(f"Unknown map type string: {value}")


class BaseMap:
    MAPS_DIR = CURRENT_DIR

    def __init__(self, map_kind: MapKind, map_type: MapType, width: int, height: int, name: str = "base_map"):
        assert width > 0, "Width should be bigger than zero"
        assert height > 0, "Height should be bigger than zero"

        self.map_kind: MapKind = map_kind
        self.map_type: MapType = map_type
        self.name: str = name
        self.size = (width, height)

        self.start_points = np.zeros((0, 2), dtype=np.float32)  # unlinked
        self.finish_points = np.zeros((0, 2), dtype=np.float32)  # unlinked
        self.linked_points = np.zeros((0, 4), dtype=np.float32)  # [start_x, start_y, finish_x, finish_y]
        self.interest_points = np.zeros((0, 2), dtype=np.float32)

    
    def update_size(self, width: int, height: int):
        raise NotImplementedError("Method needs custom implementation.")

    def save(self):
        raise NotImplementedError("Method needs custom implementation.")

    def _save(self, path: str = None, **kwargs):
        """
        Saves the map to a .npz file with the given name in a directory.

        Raises:
            AssertionError if the map is a template. It is meant to be saved from the `TemplateCollection`.
        """
        assert self.map_type != MapType.TEMPLATE, "Template maps are meant to be saved from the `TemplateCollection`"

        path = path or MAPS_FILES_DIR
        file_path = os.path.join(path, f"{self.name}.npz")
        np.savez_compressed(
            file_path,
            type=np.array([self.map_kind.value, self.map_type.value]),  # Kind, type
            size=np.array(self.size),
            interest_points=self.interest_points,
            linked_points=self.linked_points,
            **kwargs
        )

    @staticmethod
    def load(file_path: str) -> "BaseMap":
        raise NotImplementedError("Method needs custom implementation.")

    @staticmethod
    def _load(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Map file not found: {file_path}")

        return np.load(file_path, allow_pickle=True)

    def check_validity(self, print_issues: bool = True) -> bool:
        raise NotImplementedError("Method needs custom implementation.")
