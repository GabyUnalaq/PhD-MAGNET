import numpy as np
from noise import pnoise2
import matplotlib.pyplot as plt

def generate_perlin_map(width, height, scale=10.0, octaves=3, seed=None):
    """Generate a map using Perlin noise."""
    if seed is not None:
        np.random.seed(seed)
        offset_x = np.random.randint(0, 10000)
        offset_y = np.random.randint(0, 10000)
    else:
        offset_x, offset_y = 0, 0
    
    grid = np.zeros((height, width))
    
    for y in range(height):
        for x in range(width):
            # Generate Perlin noise value (-1 to 1 range typically)
            noise_val = pnoise2(
                (x + offset_x) / scale,
                (y + offset_y) / scale,
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0
            )
            grid[y, x] = noise_val
    
    return grid

def threshold_map(noise_grid, threshold=0.0):
    """Convert continuous noise to binary map (0=free, 1=obstacle)."""
    return (noise_grid > threshold).astype(int)

def cellular_automata_smooth(binary_map, iterations=3, birth_limit=4, death_limit=3):
    """
    Smooth the map using cellular automata rules.
    This is the 'smoothing based on neighbors' you mentioned!
    
    Rules (like Conway's Game of Life):
    - If a free cell (0) has >= birth_limit obstacle neighbors, it becomes obstacle
    - If an obstacle (1) has <= death_limit obstacle neighbors, it becomes free
    """
    grid = binary_map.copy()
    height, width = grid.shape
    
    for _ in range(iterations):
        new_grid = grid.copy()
        
        for y in range(height):
            for x in range(width):
                # Count obstacle neighbors (including diagonals)
                obstacle_count = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        ny, nx = y + dy, x + dx
                        
                        # Treat edges as obstacles (keeps caves away from borders)
                        if ny < 0 or ny >= height or nx < 0 or nx >= width:
                            obstacle_count += 1
                        elif grid[ny, nx] == 1:
                            obstacle_count += 1
                
                # Apply cellular automata rules
                if grid[y, x] == 1:  # Currently obstacle
                    if obstacle_count <= death_limit:
                        new_grid[y, x] = 0  # Die (become free)
                else:  # Currently free
                    if obstacle_count >= birth_limit:
                        new_grid[y, x] = 1  # Birth (become obstacle)
        
        grid = new_grid
    
    return grid

def visualize_stages(noise_grid, binary_map, smoothed_map):
    """Visualize the three stages of generation."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(noise_grid, cmap='viridis')
    axes[0].set_title('1. Perlin Noise (continuous)')
    axes[0].axis('off')
    
    axes[1].imshow(binary_map, cmap='gray_r')
    axes[1].set_title('2. After Threshold (binary)')
    axes[1].axis('off')
    
    axes[2].imshow(smoothed_map, cmap='gray_r')
    axes[2].set_title('3. After Cellular Automata (smooth)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate map
    width, height = 50, 50

    # Settings
    scale = 5.0  # Lower = more detailed caves, higher = larger open areas
    threshold = 0.05  # Controls obstacle density before smoothing: Lower threshold = more obstacles, higher = more free space
    iterations = 4  # More = smoother caves
    birth_limit = 4  # Control how aggressive the smoothing is: Higher = fewer new obstacles
    death_limit = 3  # Control how aggressive the smoothing is: Lower = obstacles survive easier
    
    # Step 1: Generate Perlin noise
    noise_grid = generate_perlin_map(width, height, scale=scale, octaves=3, seed=np.random.randint(0, 10000))
    
    # Step 2: Threshold to binary
    binary_map = threshold_map(noise_grid, threshold=threshold)
    
    # Step 3: Smooth with cellular automata
    smoothed_map = cellular_automata_smooth(
        binary_map, 
        iterations=iterations,
        birth_limit=birth_limit,
        death_limit=death_limit
    )
    
    # Visualize all stages
    visualize_stages(noise_grid, binary_map, smoothed_map)
    
    print(f"Obstacle density: {smoothed_map.sum() / smoothed_map.size * 100:.1f}%")