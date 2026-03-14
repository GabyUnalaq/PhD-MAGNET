import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

LABYRINTH_GENERATOR = "https://labyrinth.hodovanl.hu/"
LABYRINTH_PATH = Path(__file__).parent / "labyrinth.png"
GRID_SIZE = 49
TILE_START = 2  # center of first tile
TILE_STEP = 5   # pixels between tile centers
GREEN = (0, 128, 0)


def extract_labyrinth(image_path: Path = LABYRINTH_PATH) -> np.ndarray:
    """Extract a 49x49 grid from the labyrinth image.

    Returns a numpy array where 1 = obstacle (green) and 0 = walkable (black).
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            y = TILE_START + row * TILE_STEP
            x = TILE_START + col * TILE_STEP
            r, g, b = pixels[y, x]
            if (r, g, b) == GREEN:
                grid[row, col] = 1

    return grid


def visualize_labyrinth(grid: np.ndarray) -> None:
    """Display the labyrinth grid. Green = obstacle, black = walkable."""
    cmap = plt.matplotlib.colors.ListedColormap(["black", "green"])
    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap=cmap, interpolation="nearest")
    plt.title("Labyrinth (green = obstacle, black = walkable)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    grid = extract_labyrinth()
    print(f"Grid shape: {grid.shape}")
    print(f"Obstacles: {grid.sum()}, Walkable: {(grid == 0).sum()}")
    visualize_labyrinth(grid)

    # Map shinanigans
    # file_path = r"C:\Personal\_Faculta\_Doctorat\PhD-MAGNET\src\map\map_files\labyrinth.npz"
    # map_dict = dict(np.load(file_path, allow_pickle=True))
    # map_dict["grid"] = grid
    # np.savez_compressed(file_path, **map_dict)

    print("Done.")
