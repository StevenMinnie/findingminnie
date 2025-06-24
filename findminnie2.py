import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from queue import PriorityQueue
from tempfile import NamedTemporaryFile
import itertools
import pathlib
import base64
import streamlit.components.v1 as components
# ---------------- CONFIG -----------------
DEFAULT_GRID   = 20
DEFAULT_TREES  = 0.15
DEFAULT_ROCKS  = 0.08
CELL_SIZE      = 32
FPS            = 4
DOG_IMG        = "dog.png"
GOAL_IMG       = "goal.png"
BARK_FILE      = "tune.mp3"     # add your bark MP3 here

# ---------------- IMAGE LOADER -----------
@st.cache_data
def load_rgba(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

dog_icon  = load_rgba(DOG_IMG , CELL_SIZE)
goal_icon = load_rgba(GOAL_IMG, CELL_SIZE)

def overlay_rgba(base, overlay, x, y):
    h, w = overlay.shape[:2]
    roi  = base[y:y+h, x:x+w]
    a = overlay[..., 3:] / 255.0
    roi[:] = (overlay[..., :3] * a + roi * (1 - a)).astype(np.uint8)

# ---------------- A* ---------------------
def heuristic(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid,start,goal):
    rows,cols = grid.shape
    pq = PriorityQueue(); pq.put((0,start))
    came,g = {},{start:0}
    while not pq.empty():
        _,cur = pq.get()
        if cur==goal:
            path=[]; 
            while cur in came: path.append(cur); cur=came[cur]
            path.append(start); return path[::-1]
        for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx,ny = cur[0]+dx,cur[1]+dy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                t=g[cur]+1
                if t<g.get((nx,ny),1e9):
                    came[(nx,ny)]=cur; g[(nx,ny)]=t
                    pq.put((t+heuristic((nx,ny),goal),(nx,ny)))
    return None

# ---------------- MAP --------------------
def make_grass_tile(size,j=12):
    h=60+np.random.randint(-j,j,(size,size))
    s=180+np.random.randint(-30,30,(size,size))
    v=140+np.random.randint(-25,25,(size,size))
    hsv=np.dstack((h,s,v)).astype(np.uint8)
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

def generate_grid_with_path(n,tree,rock):
    for _ in range(1000):
        grid=np.zeros((n,n),np.uint8)
        for _ in range(int(n*n*tree)):
            grid[np.random.randint(n),np.random.randint(n)]=1
        idx=np.random.choice(n*n,int(n*n*rock),False)
        grid[np.unravel_index(idx,grid.shape)]=1
        for _ in range(2):
            c=np.random.randint(4,n-4); gap=np.random.randint(2,n-2)
            for r in range(n):
                if r!=gap: grid[r,c]=1
        grid[0,0]=grid[-1,-1]=0
        path=astar(grid,(0,0),(n-1,n-1))
        if path: return grid,path
    return None,None

# ---------------- FRAME ------------------
def draw_frame(grid,path_seg,obs,cell=CELL_SIZE):
    n=grid.shape[0]
    grass=make_grass_tile(cell)
    frame=np.zeros((n*cell,n*cell,3),np.uint8)
    for r,c in itertools.product(range(n),range(n)):
        frame[r*cell:(r+1)*cell, c*cell:(c+1)*cell]=grass
        if grid[r,c]:
            if obs[r,c]<.75:
                cv2.circle(frame,(c*cell+cell//2,r*cell+cell//2),cell//2-2,(0,70,0),-1)
            else:
                cv2.rectangle(frame,(c*cell+2,r*cell+2),((c+1)*cell-2,(r+1)*cell-2),(105,105,105),-1)
    for r,c in path_seg:
        cv2.rectangle(frame,(c*cell,r*cell),((c+1)*cell-1,(r+1)*cell-1),(180,255,180),-1)
    overlay_rgba(frame,goal_icon,(n-1)*cell,(n-1)*cell)
    dr,dc=path_seg[-1]; overlay_rgba(frame,dog_icon,dc*cell,dr*cell)
    return frame

# ---------------- STREAMLIT UI -----------
st.set_page_config("Find Minnie", layout="centered")
st.markdown("""
<style>
[data-testid="stSidebar"][aria-expanded="false"]{width:300px;margin-left:0;}
</style>
""", unsafe_allow_html=True)
st.title("🐶 Finding Princess Minnie's Lap – An Interactive A* Algorithm Demo")
st.markdown("""
### 🔧 How to Use:
1. Use the sliders in the sidebar (click **≫** in the top-left if it's hidden).
2. Select the values of the map size, the density of obstacles such as trees and rocks. 
3. The A* algorithm will always find a path if it exists. If such a path doesn't exist, you'll be requested to lower either the map density or the densities of obstacles. 
4. Click **Generate New Map & Solve** to create a map and see the puppy find Princess Minnie.
5. A GIF animation will be shown automatically.
6. Turn on your audio.
7. You can also download the animation by clicking **Download GIF**.

Enjoy the adventure! 🐾
""")


with st.sidebar:
    st.header("Map Parameters")
    N      = st.slider("Grid size (N×N)",10,40,DEFAULT_GRID)
    T_R    = st.slider("Tree density",0.0,0.4,DEFAULT_TREES,0.01)
    R_R    = st.slider("Rock density",0.0,0.4,DEFAULT_ROCKS,0.01)

if st.button("Generate New Map & Solve"):
    grid,path = generate_grid_with_path(N,T_R,R_R)
    if path is None:
        st.error("No solvable map found. Try lower densities.")
        st.stop()

    obs_types = np.random.rand(N,N)
    st.image(draw_frame(grid,path,obs_types),channels="BGR",
             caption=f"Path length: {len(path)-1} steps")

    prog=st.progress(0,"Rendering GIF …")
    fig,ax=plt.subplots(figsize=(6,6)); plt.axis("off")
    def update(i):
        ax.clear(); ax.axis("off")
        ax.imshow(cv2.cvtColor(draw_frame(grid,path[:i+1],obs_types),cv2.COLOR_BGR2RGB))
    anim=FuncAnimation(fig,update,frames=len(path),interval=600)

    with NamedTemporaryFile(suffix=".gif",delete=False) as tmp:
        anim.save(tmp.name,writer=PillowWriter(fps=FPS))
        prog.progress(100,"Done!")
        st.image(tmp.name)
        st.download_button("Download GIF",open(tmp.name,"rb").read(),
                           "find_minnie.gif","image/gif")

    # ---------- Bark sound ---------------
    if pathlib.Path(BARK_FILE).is_file():
        b64_audio = base64.b64encode(open(BARK_FILE, "rb").read()).decode()
        components.html(f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
        """, height=0)
    else:
        st.info("🔈 Add a bark.mp3 file to play a woof when Minnie is found!")

else:
    st.info("Click **Generate New Map & Solve** to create a maze.")



