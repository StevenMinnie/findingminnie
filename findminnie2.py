import streamlit as st, numpy as np, cv2, matplotlib.pyplot as plt, itertools, time, base64, pathlib, streamlit.components.v1 as components
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
    h, w = o.shape[:2]; a = o[..., 3:] / 255.0
    roi = b[y:y+h, x:x+w]
    roi[:] = (o[..., :3] * a + roi * (1 - a)).astype(np.uint8)

def manh(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

def search(g, s, t, alg):
    pq, came, cost = PriorityQueue(), {}, {s: 0}
    pq.put((0, 0, s)); explored = 0
    while not pq.empty():
        _, g_cost, cur = pq.get(); explored += 1
        if cur == t:
            p = [];  c = cur
            while c in came: p.append(c); c = came[c]
            p.append(s); return p[::-1], explored
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not (0 <= nx < g.shape[0] and 0 <= ny < g.shape[1]) or g[nx, ny]: continue
            ncost = g_cost + 1
            if ncost < cost.get((nx, ny), 1e9):
                cost[(nx, ny)] = ncost; came[(nx, ny)] = cur
                h = 0 if alg == "Dijkstra" else manh((nx, ny), t)
                f = ncost + h if alg != "Greedy" else h
                pq.put((f, ncost, (nx, ny)))
    return None, explored

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
    n = g.shape[0]; f = np.zeros((n*CELL, n*CELL, 3), np.uint8); tile = grass_tile(CELL)
    for r,c in itertools.product(range(n), range(n)):
        f[r*CELL:(r+1)*CELL, c*CELL:(c+1)*CELL] = tile
        if g[r,c]:
            if obs[r,c] < .75: cv2.circle(f,(c*CELL+CELL//2,r*CELL+CELL//2),CELL//2-2,(0,70,0),-1)
            else: cv2.rectangle(f,(c*CELL+2,r*CELL+2),((c+1)*CELL-2,(r+1)*CELL-2),(105,105,105),-1)
    for r,c in seg: cv2.rectangle(f,(c*CELL,r*CELL),((c+1)*CELL-1,(r+1)*CELL-1),(180,255,180),-1)
    overlay_rgba(f, goal_icon, (n-1)*CELL, (n-1)*CELL)
    dr, dc = seg[-1]; overlay_rgba(f, dog_icon, dc*CELL, dr*CELL)
    return f

st.set_page_config("Find Minnie", layout="centered")
st.markdown('<style>[data-testid="stSidebar"][aria-expanded="false"]{width:300px;margin-left:0}</style>', unsafe_allow_html=True)
st.title("🐶 Finding Princess Minnie's Lap – An Interactive Path-Finding Demo")


st.markdown("""
### 🔧 How to Use:
1. Open the sidebar sliders (click **≫** if hidden).  
2. Pick map size, tree and rock densities, and an algorithm (A*, Djikstra's, or Greedy). 
3. Click **Generate / Regenerate Map** to create a map.  
4. Switch algorithms to compare paths on the **same map**. Don't click on **Generate / Regenerate Map** if you're only switching algorithms.  
5. Watch the GIF and keep the audio on.  
6. Download the GIF if you like.
7. The performance of each algorithm using metrics such as distance, nodes expanded, and total time taken is displayed below for the particular map in use. 

**Note:** If an algorithm fails to find a path, either try a different one or regenerate the map with lower grid size or obstacle densities. You will know a particular algorithm has failed when you generate a new map and it is taking a long time to load it on the screen. 

Enjoy the adventure! 🐾
- Created By Minnie's Puppy
""")

with st.sidebar:
    N  = st.slider("Grid size", 10, 40, DEFAULT_GRID)
    TR = st.slider("Tree density", 0.0, 0.4, DEFAULT_TREES, 0.01)
    RR = st.slider("Rock density", 0.0, 0.4, DEFAULT_ROCKS, 0.01)
    algo = st.selectbox("Algorithm to animate", ["A*", "Dijkstra", "Greedy"])
    regen = st.button("Generate / Regenerate Map")

if regen or "grid" not in st.session_state:
    while True:
        g = gen_grid(N, TR, RR)
        p,_ = search(g,(0,0),(N-1,N-1),"A*")
        if p: 
            st.session_state.grid = g
            st.session_state.obs  = np.random.rand(N,N)
            st.session_state.metrics = {}
            break
    st.success("New map ready. Metrics updated.")

if "grid" in st.session_state:
    if not st.session_state.metrics:
        for m in ["A*", "Dijkstra", "Greedy"]:
            t0 = time.perf_counter()
            path, explored = search(st.session_state.grid, (0,0), (N-1,N-1), m)
            t1 = time.perf_counter()
            st.session_state.metrics[m] = {"Path": path, "Length": len(path)-1 if path else "-", "Explored": explored, "Time": (t1-t0)*1000}
    st.table([{ "Algorithm": k, 
                "Path length": v["Length"], 
                "Explored nodes": v["Explored"], 
                "Time (ms)": f"{v['Time']:.2f}" if v["Path"] else "-" } 
              for k,v in st.session_state.metrics.items()])

    data = st.session_state.metrics.get(algo)
    if data and data["Path"]:
        st.image(frame(st.session_state.grid, data["Path"], st.session_state.obs), channels="BGR", caption=f"{algo} length {data['Length']}")
        prog = st.progress(0,"GIF …"); fig, ax = plt.subplots(figsize=(6,6)); plt.axis("off")
        def upd(i):
            ax.clear(); ax.axis("off")
            ax.imshow(cv2.cvtColor(frame(st.session_state.grid,data["Path"][:i+1],st.session_state.obs),cv2.COLOR_BGR2RGB))
        anim = FuncAnimation(fig, upd, frames=len(data["Path"]), interval=600)
        with NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
            anim.save(tmp.name, writer=PillowWriter(fps=FPS))
            prog.progress(100,"Done!"); st.image(tmp.name)
            st.download_button("Download GIF", open(tmp.name,"rb").read(), "find_minnie.gif","image/gif")
        if pathlib.Path(BARK).is_file():
            b64 = base64.b64encode(open(BARK,"rb").read()).decode()
            components.html(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', height=0)
    else:
        st.warning(f"{algo} could not find a path; switch algorithm or regenerate map.")




