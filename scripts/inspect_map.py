import sys
import os
import numpy as np
from src.map._base_map import MapType, MapKind
from src.map import MAPS_FILES_DIR

def load_templates(): ...


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.map.inspect <type> <name>")
        print("  type: 'map' or 'template'")
        print("  name: file name without extension (e.g. grid_map)")
        sys.exit(1)

    if sys.argv[1] != "map":
        raise NotImplementedError("Template inspect not yet implemented.")

    name = sys.argv[2]
    file_path = os.path.join(MAPS_FILES_DIR, f"{name}.npz")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    data = np.load(file_path, allow_pickle=True)
    map_kind, map_type = data["type"]
    map_type = MapType(int(map_type))
    map_kind = MapKind(int(map_kind))

    if map_type == MapType.TEMPLATE:
        raise NotImplementedError("Template inspect not yet implemented.")

    print(f"File: {file_path}")
    print(f"Size: {data['size']}")
    print(f"Kind: {map_kind.name}")
    print(f"Fields: {list(data.keys())}")
    if "interest_points" in data:
        print(f"Interest Points ({len(data['interest_points'])})")
    if "linked_points" in data:
        print(f"Linked Points ({len(data['linked_points'])})")
    

    if map_kind == MapKind.GRID:
        grid = data["grid"]
        print(f"Obstacles: {grid.sum()}, Walkable: {(grid == 0).sum()}")
        # print(f"\nGrid Data:\n{data['grid']}")


if __name__ == "__main__":
    main()
