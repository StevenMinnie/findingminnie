import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from queue import PriorityQueue
from tempfile import NamedTemporaryFile
from PIL import Image
import itertools

# --- Configuration ---
DEFAULT_GRID   = 20
DEFAULT_TREES  = 0.15
DEFAULT_ROCKS  = 0.08
CELL_SIZE      = 32
FPS            = 4
DOG_IMG        = "dog.png"
GOAL_IMG       = "goal.png"

@st.cache_data
def load_rgba(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img

dog_icon  = load_rgba(DOG_IMG,  CELL_SIZE)
goal_icon = load_rgba(GOAL_IMG, CELL_SIZE)

def overlay_rgba(base, overlay, x, y):
    h, w = overlay.shape[:2]
    roi  = base[y:y+h, x:x+w]
    alpha = overlay[..., 3:] / 255.0
    roi[:] = (overlay[..., :3] * alpha + roi * (1 - alpha)).astype(np.uint8)

# --- A* ---
def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    pq = PriorityQueue(); pq.put((0, start))
    came, gscore = {}, {start: 0}
    while not pq.empty():
        _, cur = pq.get()
        if cur == goal:
            path = []
            while cur in came: path.append(cur); cur = came[cur]
            path.append(start); return path[::-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                tentative = gscore[cur] + 1
                if tentative < gscore.get((nx,ny),1e9):
                    came[(nx,ny)] = cur
                    gscore[(nx,ny)] = tentative
                    pq.put((tentative + heuristic((nx,ny), goal), (nx,ny)))
    return None

# --- Map ---
def make_grass_tile(size, jitter=12):
    h = 60 + np.random.randint(-jitter,jitter,(size,size))
    s = 180 + np.random.randint(-30,30,(size,size))
    v = 140 + np.random.randint(-25,25,(size,size))
    hsv = np.dstack((h,s,v)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def generate_grid_with_path(n, tree_ratio, rock_ratio):
    for _ in range(1000):
        grid = np.zeros((n,n), np.uint8)
        for _ in range(int(n*n*tree_ratio)):
            x,y = np.random.randint(n), np.random.randint(n)
            grid[x,y]=1
        idx = np.random.choice(n*n, int(n*n*rock_ratio), False)
        grid[np.unravel_index(idx, grid.shape)] = 1
        for _ in range(2):
            col = np.random.randint(4,n-4); gap = np.random.randint(2,n-2)
            for r in range(n):
                if r!=gap: grid[r,col]=1
        grid[0,0]=grid[-1,-1]=0
        path = astar(grid,(0,0),(n-1,n-1))
        if path: return grid,path
    return None,None

# --- Frame drawing ---
def draw_frame(grid, path_segment, obstacle_types, cell=CELL_SIZE):
    n = grid.shape[0]
    grass = make_grass_tile(cell)
    frame = np.zeros((n*cell, n*cell, 3), np.uint8)
    for r,c in itertools.product(range(n), range(n)):
        frame[r*cell:(r+1)*cell, c*cell:(c+1)*cell] = grass
    for r,c in itertools.product(range(n), range(n)):
        if grid[r,c]:
            if obstacle_types[r,c] < .75:
                cv2.circle(frame,(c*cell+cell//2,r*cell+cell//2),
                           cell//2-2,(0,70,0),-1)
            else:
                cv2.rectangle(frame,(c*cell+2,r*cell+2),
                              ((c+1)*cell-2,(r+1)*cell-2),(105,105,105),-1)
    for r,c in path_segment:
        cv2.rectangle(frame,(c*cell,r*cell),
                      ((c+1)*cell-1,(r+1)*cell-1),(180,255,180),-1)
    overlay_rgba(frame, goal_icon, (n-1)*cell, (n-1)*cell)
    dog_r, dog_c = path_segment[-1]
    overlay_rgba(frame, dog_icon, dog_c*cell, dog_r*cell)
    return frame

# --- UI ---
st.set_page_config("Find Minnie", layout="centered")
st.title("🐶 Finding Princess Minnie's Lap – An Interactive A* Algorithm Path Demo. Please click on the double arrows on the top left corner to access the sliders.")

with st.sidebar:
    st.header("Map Parameters")
    N      = st.slider("Grid size (N×N)", 10, 40, DEFAULT_GRID )
    T_RATIO = st.slider("Tree density",  0.00, 0.40, DEFAULT_TREES, 0.01 )
    R_RATIO = st.slider("Rock density",  0.00, 0.40, DEFAULT_ROCKS, 0.01 )

if st.button("Generate New Map & Solve"):
    grid, path = generate_grid_with_path(N, T_RATIO, R_RATIO)
    if path is None:
        st.error("Could not find a solvable map after many tries. Lower densities?")
        st.stop()

    obstacle_types = np.random.rand(N, N)

    frame = draw_frame(grid, path, obstacle_types, CELL_SIZE)
    st.image(frame, caption=f"Path length: {len(path)-1} steps", channels="BGR")

    prog = st.progress(0, "Rendering GIF …")
    fig, ax = plt.subplots(figsize=(6,6)); plt.axis("off")

    def draw(step):
        ax.clear(); ax.axis("off")
        frame = draw_frame(grid, path[:step+1], obstacle_types, CELL_SIZE)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    anim = FuncAnimation(fig, draw, frames=len(path), interval=600)
    with NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        anim.save(tmp.name, writer=PillowWriter(fps=FPS))
        prog.progress(100, "Done!")
        st.image(tmp.name)
        st.download_button("Download GIF", open(tmp.name,"rb").read(),
                           file_name="find_minnie.gif", mime="image/gif")
else:
    st.info("Click **Generate New Map & Solve** to create a maze.")



