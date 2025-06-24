import streamlit as st, numpy as np, cv2, matplotlib.pyplot as plt, itertools, base64, pathlib, streamlit.components.v1 as components
from matplotlib.animation import FuncAnimation, PillowWriter
from queue import PriorityQueue
from tempfile import NamedTemporaryFile

DEFAULT_GRID, DEFAULT_TREES, DEFAULT_ROCKS = 20, 0.15, 0.08
CELL, FPS = 32, 4
DOG_IMG, GOAL_IMG, BARK = "dog.png", "goal.png", "tune.mp3"

@st.cache_data
def load_rgba(p, s):
    i = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if i is None: raise FileNotFoundError(p)
    i = cv2.cvtColor(i, cv2.COLOR_BGR2RGBA) if i.shape[2] == 3 else cv2.cvtColor(i, cv2.COLOR_BGRA2RGBA)
    return cv2.resize(i, (s, s), interpolation=cv2.INTER_AREA)

dog_icon, goal_icon = load_rgba(DOG_IMG, CELL), load_rgba(GOAL_IMG, CELL)

def overlay_rgba(b, o, x, y):
    h, w = o.shape[:2]
    r = b[y:y+h, x:x+w]
    a = o[..., 3:] / 255.0
    r[:] = (o[..., :3] * a + r * (1 - a)).astype(np.uint8)

def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def search(g, s, t, alg):
    pq, came, cost = PriorityQueue(), {}, {s: 0}
    pq.put((0, 0, s))
    while not pq.empty():
        _, cst, cur = pq.get()
        if cur == t:
            p = []
            while cur in came: p.append(cur); cur = came[cur]
            p.append(s); return p[::-1]
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0 <= nx < g.shape[0] and 0 <= ny < g.shape[1]) or g[nx, ny]: continue
            ncost = cst + 1
            if ncost < cost.get((nx, ny), 1e9):
                cost[(nx, ny)] = ncost
                came[(nx, ny)] = cur
                hh = 0 if alg == "Dijkstra" else h((nx, ny), t)
                f = ncost + hh if alg != "Greedy" else hh
                pq.put((f, ncost, (nx, ny)))
    return None

def grass_tile(sz):
    hsv = np.dstack((60+np.random.randint(-12,12,(sz,sz)),
                     180+np.random.randint(-30,30,(sz,sz)),
                     140+np.random.randint(-25,25,(sz,sz)))).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def gen_grid(n, tr, rr):
    g = np.zeros((n,n), np.uint8)
    for _ in range(int(n*n*tr)): g[np.random.randint(n), np.random.randint(n)] = 1
    idx = np.random.choice(n*n, int(n*n*rr), False)
    g[np.unravel_index(idx, g.shape)] = 1
    for _ in range(2):
        c, gap = np.random.randint(4, n-4), np.random.randint(2, n-2)
        for r in range(n):
            if r != gap: g[r,c] = 1
    g[0,0] = g[-1,-1] = 0
    return g

def frame(g, seg, obs):
    n = g.shape[0]
    f = np.zeros((n*CELL, n*CELL, 3), np.uint8)
    tile = grass_tile(CELL)
    for r,c in itertools.product(range(n), range(n)):
        f[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL] = tile
        if g[r,c]:
            if obs[r,c] < .75: cv2.circle(f,(c*CELL+CELL//2,r*CELL+CELL//2), CELL//2-2, (0,70,0), -1)
            else: cv2.rectangle(f,(c*CELL+2,r*CELL+2),((c+1)*CELL-2,(r+1)*CELL-2),(105,105,105),-1)
    for r,c in seg: cv2.rectangle(f,(c*CELL,r*CELL),((c+1)*CELL-1,(r+1)*CELL-1),(180,255,180),-1)
    overlay_rgba(f, goal_icon, (n-1)*CELL, (n-1)*CELL)
    dr, dc = seg[-1]
    overlay_rgba(f, dog_icon, dc*CELL, dr*CELL)
    return f

if "grid" not in st.session_state: st.session_state.grid = None
if "obs"  not in st.session_state: st.session_state.obs  = None

st.set_page_config("Find Minnie", layout="centered")
st.markdown('<style>[data-testid="stSidebar"][aria-expanded="false"]{width:300px;margin-left:0;}</style>', unsafe_allow_html=True)
st.title("🐶 Finding Princess Minnie's Lap – An Interactive Pathfinding Demo")
st.markdown("""
### 🔧 How to Use:
1. Open the sidebar sliders (click **≫** if hidden).  
2. Pick map size, tree and rock densities, and an algorithm.  
3. Click **Generate / Regenerate Map** to create a map.  
4. Switch algorithms to compare paths on the **same map**. Don't click on **Generate / Regenerate Map** if you're only switching algorithms. 
5. Watch the GIF and keep the audio on.  
6. Download the GIF if you like.  

Note: Everytike you change the parameters such as grid size or the obstacle densities, you need to click on **Generate / Regenerate Map**. It is possible that a case might arise where a particular algorithm can't find a path to the goal. In such a case, please either switch the algorithm or change the parameters.

Enjoy the adventure! 🐾
""")

with st.sidebar:
    N = st.slider("Grid size", 10, 40, DEFAULT_GRID)
    T_R = st.slider("Tree density", 0.0, 0.4, DEFAULT_TREES, 0.01)
    R_R = st.slider("Rock density", 0.0, 0.4, DEFAULT_ROCKS, 0.01)
    algo = st.selectbox("Algorithm", ["A*", "Dijkstra", "Greedy"])
    regenerate = st.button("Generate / Regenerate Map")

if regenerate:
    while True:
        g = gen_grid(N, T_R, R_R)
        if search(g, (0,0), (N-1, N-1), "A*"):
            st.session_state.grid = g
            st.session_state.obs  = np.random.rand(N, N)
            break
    st.success("New map generated. Now choose an algorithm to visualize.")

if st.session_state.grid is not None:
    path = search(st.session_state.grid, (0,0), (N-1, N-1), algo)
    if path:
        st.image(frame(st.session_state.grid, path, st.session_state.obs), channels="BGR", caption=f"{algo} path length {len(path)-1}")
        prog = st.progress(0, "Rendering GIF …")
        fig, ax = plt.subplots(figsize=(6,6)); plt.axis("off")
        def upd(i):
            ax.clear(); ax.axis("off")
            ax.imshow(cv2.cvtColor(frame(st.session_state.grid, path[:i+1], st.session_state.obs), cv2.COLOR_BGR2RGB))
        anim = FuncAnimation(fig, upd, frames=len(path), interval=600)
        with NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            anim.save(tmp.name, writer=PillowWriter(fps=FPS))
            prog.progress(100, "Done!")
            st.image(tmp.name)
            st.download_button("Download GIF", open(tmp.name,"rb").read(), "find_minnie.gif", "image/gif")
        if pathlib.Path(BARK).is_file():
            b64 = base64.b64encode(open(BARK,"rb").read()).decode()
            components.html(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', height=0)
    else:
        st.warning("Selected algorithm failed to find a path on this map. Try another algorithm or regenerate the map.")





