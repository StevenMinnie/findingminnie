import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from queue import PriorityQueue
import os

dog_img_path = "dog.png"     
goal_img_path = "goal.png"   
output_path = "a_star_dog_path_obstacle_maze.mp4"

grid_size = (20, 20)
cell_size = 32
fps = 4
grass_hue_jit = 12
tree_ratio = 0.15
rock_ratio = 0.08


def load_image_rgba(path, target_sz):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    img = cv2.resize(img, target_sz, interpolation=cv2.INTER_AREA)
    return img

dog_icon = load_image_rgba(dog_img_path, (cell_size, cell_size))
goal_icon = load_image_rgba(goal_img_path, (cell_size, cell_size))

def make_grass_tile(size, hue_jit=12):
    h = 60 + np.random.randint(-hue_jit, hue_jit, (size, size))
    s = 180 + np.random.randint(-30, 30, (size, size))
    v = 140 + np.random.randint(-25, 25, (size, size))
    hsv = np.dstack((h, s, v)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

grass_tile = make_grass_tile(cell_size, grass_hue_jit)


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    while not open_set.empty():
        _, current = open_set.get()
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] == 0:
                tentative = g_score[current] + 1
                if tentative < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative
                    fscore = tentative + heuristic((nx, ny), goal)
                    open_set.put((fscore, (nx, ny)))
    return None


def generate_grid_with_path():
    rows, cols = grid_size
    for _ in range(1000):
        grid = np.zeros((rows, cols), dtype=np.uint8)

        # Scattered trees
        for _ in range(int(rows * cols * tree_ratio)):
            x, y = np.random.randint(rows), np.random.randint(cols)
            grid[x, y] = 1

        # Random rocks
        rock_cells = int(rows * cols * rock_ratio)
        rock_idx = np.random.choice(rows*cols, rock_cells, replace=False)
        grid[np.unravel_index(rock_idx, grid.shape)] = 1

        # Vertical walls with gaps
        for _ in range(2):
            wall_col = np.random.randint(4, cols - 4)
            gap_row = np.random.randint(2, rows - 2)
            for r in range(rows):
                if r != gap_row:
                    grid[r, wall_col] = 1

        grid[0, 0] = grid[-1, -1] = 0

        path = astar(grid, (0, 0), (rows-1, cols-1))
        if path:
            return grid, path
    raise RuntimeError("Failed to generate a valid grid with path.")

grid, path = generate_grid_with_path()
canvas_h, canvas_w = grid_size[0]*cell_size, grid_size[1]*cell_size
obstacle_types = np.random.rand(*grid.shape)


def overlay_rgba(base, overlay, top_left):
    y, x = top_left
    h, w = overlay.shape[:2]
    roi = base[y:y+h, x:x+w]
    alpha = overlay[...,3:4] / 255.0
    inv_alpha = 1.0 - alpha
    overlay_rgb = overlay[...,:3].astype(float)
    roi[:] = (alpha * overlay_rgb + inv_alpha * roi.astype(float)).astype(np.uint8)


def draw_frame(step):
    frame = np.zeros((canvas_h, canvas_w, 3), np.uint8)

    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            y, x = r*cell_size, c*cell_size
            frame[y:y+cell_size, x:x+cell_size] = grass_tile

    for r in range(grid_size[0]):
        for c in range(grid_size[1]):
            if grid[r, c]:
                if obstacle_types[r, c] < 0.75:
                    cv2.circle(frame, (c*cell_size + cell_size//2, r*cell_size + cell_size//2),
                               cell_size//2 - 2, (0, 70, 0), -1)
                else:
                    cv2.rectangle(frame, (c*cell_size+2, r*cell_size+2),
                                  ((c+1)*cell_size-2, (r+1)*cell_size-2), (105,105,105), -1)

    for (r, c) in path[:step+1]:
        cv2.rectangle(frame, (c*cell_size, r*cell_size),
                      ((c+1)*cell_size-1, (r+1)*cell_size-1), (180,255,180), -1)

    overlay_rgba(frame, goal_icon, ((grid_size[0]-1)*cell_size, (grid_size[1]-1)*cell_size))

    pos_r, pos_c = path[min(step, len(path)-1)]
    overlay_rgba(frame, dog_icon, (pos_r*cell_size, pos_c*cell_size))

    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.axis('off')


fig, ax = plt.subplots(figsize=(6,6))
plt.axis('off')
ani = FuncAnimation(fig, draw_frame, frames=len(path), interval=600)
ani.save(output_path, writer='ffmpeg', fps=fps)

print("Animation saved to:", output_path)

