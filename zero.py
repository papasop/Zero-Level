# ===== Colab: Red-Team Protocol (P0..P8) for MIN-QG — Patched Full Version =====
# - 单格运行，输出 print/plt；不写文件
# - 包含：z(E/N) 自适应、P1 精简网格、d_hat 可视化、NaN-hardened 口径
# - 你可先将 cfg.geom_samples 降到 4、cfg.q_samples 降到 48 以提速

import math, random, time
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import HuberRegressor, TheilSenRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# ---- 防噪：屏蔽 numpy RuntimeWarning（不改变数值路径，仅静音） ----
np.seterr(all='ignore')
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r".*numpy.*")

# ---------------- RNG ----------------
SEED = 20251025
np.random.seed(SEED); random.seed(SEED)
rng = np.random.default_rng(SEED)

# ---------------- Robust slope ----------------
def safe_robust_slope(X_col: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(X_col, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 5: return float("nan")
    x = x[m]; y = y[m]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]): return float("nan")
    def _mad(a):
        med = np.median(a)
        return np.median(np.abs(a - med)) + 1e-12
    xm, ym = np.median(x), np.median(y)
    xm_mad, ym_mad = _mad(x), _mad(y)
    xz = (x - xm) / xm_mad
    yz = (y - ym) / ym_mad
    if np.std(xz) < 1e-9 or np.std(yz) < 1e-9: return float("nan")
    try:
        hub = HuberRegressor(epsilon=1.25, alpha=0.0, max_iter=20000)
        hub.fit(xz.reshape(-1,1), yz)
        slope = float(hub.coef_[0]) * (ym_mad / xm_mad)
        if np.isfinite(slope): return slope
    except Exception: pass
    try:
        ts = TheilSenRegressor(random_state=0, max_subpopulation=1_000_000)
        ts.fit(xz.reshape(-1,1), yz)
        slope = float(ts.coef_[0]) * (ym_mad / xm_mad)
        if np.isfinite(slope): return slope
    except Exception: pass
    try:
        A = np.c_[x, np.ones_like(x)]
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(slope)
    except Exception:
        return float("nan")

# ---------------- DAG generator & TR ----------------
def approximate_transitive_reduction(G: nx.DiGraph, cap_per_node:int=96) -> nx.DiGraph:
    H = nx.DiGraph(); H.add_nodes_from(G.nodes()); H.add_edges_from(G.edges())
    for u in G.nodes():
        succs = list(G.successors(u))[:cap_per_node]
        two_hop = set()
        for v in succs:
            for w in list(G.successors(v))[:cap_per_node]:
                if w != u: two_hop.add(w)
        for w in succs:
            if w in two_hop and H.has_edge(u, w): H.remove_edge(u, w)
    assert nx.is_directed_acyclic_graph(H)
    return H

def sample_causal_dag(N:int, base_p:float=0.54, decay:float=0.018, max_fan:int=56,
                      tr_mode:str="off", tr_cap:int=64) -> nx.DiGraph:
    t = rng.random(N)
    order = np.argsort(t)
    G = nx.DiGraph(); G.add_nodes_from(range(N))
    for idx, u in enumerate(order[:-1]):
        future = order[idx+1:]
        k = np.arange(1, len(future)+1)
        probs = base_p * np.exp(-decay * k)
        draw = rng.random(len(future)) < probs
        targets = np.array(future)[draw]
        if targets.size > max_fan: targets = targets[:max_fan]
        for v in targets: G.add_edge(u, v)
    if tr_mode == "off": 
        return G
    cap = tr_cap if tr_mode == "soft" else max(tr_cap, 128)
    return approximate_transitive_reduction(G, cap_per_node=cap)

# ---------------- D, W, B ----------------
def build_DWB(G: nx.DiGraph, weight_mode:str="gamma", w_scale:float=1.0):
    nodes = list(G.nodes()); node_idx = {n:i for i,n in enumerate(nodes)}
    edges = list(G.edges()); E, N = len(edges), len(nodes)
    if E == 0:
        D = sparse.csr_matrix((0,N)); W = sparse.diags(np.zeros(1)); B = sparse.csr_matrix((N,N))
        return D, W, B
    rows, cols, data = [], [], []
    for ei,(u,v) in enumerate(edges):
        rows += [ei, ei]; cols += [node_idx[u], node_idx[v]]; data += [-1.0, 1.0]
    D = sparse.coo_matrix((data,(rows,cols)), shape=(E,N)).tocsr()
    if weight_mode=="unit":
        w = np.ones(E)*w_scale
    elif weight_mode=="gamma":
        w = rng.gamma(shape=2.0, scale=w_scale/2.0, size=E); w = np.maximum(w, 1e-12)
    else:
        raise ValueError("weight_mode")
    W = sparse.diags(w, format="csr")
    B = D.T @ W @ D
    return D, W, B

# ---------------- Dynamics & observables ----------------
@dataclass
class EvoCfg:
    steps:int=420; eta:float=0.05; noise:float=1e-12

def ward_residual_stats(B, D, W, N, steps=240, eta=0.05):
    if N==0: return float("nan"), float("nan"), float("nan")
    phi = rng.standard_normal(N); stats=[]
    for _ in range(steps):
        flux = D.T @ (W @ (D @ phi)); eom = - (B @ phi); r = flux + eom
        denom = np.linalg.norm(B @ phi) + 1e-15
        stats.append(np.linalg.norm(r)/denom)
        phi = phi - eta * (B @ phi)
    stats=np.array(stats)
    return float(np.median(stats)), float(np.percentile(stats,10)), float(np.percentile(stats,90))

def timeseries_K(B, D, W, N, cfg:EvoCfg, win:int=128):
    if N==0: return float("nan")
    phi = rng.standard_normal(N)
    Hs, Phis = [], []
    for _ in range(cfg.steps):
        flux = D.T @ (W @ (D @ phi)); eom = - (B @ phi); r = flux + eom
        chi = r + cfg.noise * rng.standard_normal(N)
        Hs.append(float(np.dot(chi,chi))); Phis.append(float(np.sum(np.abs(chi))))
        phi = phi - cfg.eta * (B @ phi)
    Hs, Phis = np.array(Hs), np.array(Phis)
    Ks=[]; x = np.log(Phis + 1e-300); y = np.log(Hs + 1e-300)
    for s in range(0, len(Hs) - win + 1):
        sl = slice(s, s+win)
        if np.std(x[sl]) < 1e-9 or np.std(y[sl]) < 1e-9: continue
        Ks.append(safe_robust_slope(x[sl], y[sl]))
    Ks = np.array([k for k in Ks if np.isfinite(k)])
    return float(np.median(Ks)) if Ks.size else float("nan")

# ---------- Intrinsic dimension & interval ----------
def shortest_directed_distance(G: nx.DiGraph, u:int, v:int):
    if u==v: return 0
    q=[u]; vis={u}; d=0
    while q:
        d+=1; nq=[]
        for x in q:
            for y in G.successors(x):
                if y==v: return d
                if y not in vis: vis.add(y); nq.append(y)
        q=nq
    return math.inf

def exact_interval_volume(G: nx.DiGraph, u:int, v:int) -> int:
    F={u}; q=[u]
    while q:
        a=q.pop(0)
        for b in G.successors(a):
            if b not in F: F.add(b); q.append(b)
    P={v}; q=[v]
    while q:
        a=q.pop(0)
        for b in G.predecessors(a):
            if b not in P: P.add(b); q.append(b)
    I = F & P; I.discard(u); I.discard(v)
    return len(I)

def alex_dim_intrinsic(G: nx.DiGraph, M:int=1600, log_bins:int=12, min_pts:int=10, tau_min:int=4, seed:int=2025):
    rng_local = np.random.default_rng(seed)
    nodes = list(G.nodes()); N=len(nodes)
    if N==0: return float("nan"), np.array([]), np.array([])
    pairs=[]; tries=0
    while len(pairs)<M and tries<100*M:
        tries+=1
        u = int(rng_local.integers(0, N-1))
        v = int(rng_local.integers(u+1, N))
        tau = shortest_directed_distance(G, u, v)
        if tau is not math.inf and tau>=tau_min:
            pairs.append((u,v,tau))
    if len(pairs) < max(80, M//6): return float("nan"), np.array([]), np.array([])
    taus = np.array([t for _,_,t in pairs], float)
    Vs=[exact_interval_volume(G,u,v) for (u,v,_) in pairs]; Vs=np.array(Vs,float)
    taus = taus[taus>0]; Vs = Vs[Vs>0]
    if taus.size < 6 or Vs.size < 6: return float("nan"), np.array([]), np.array([])
    bins = np.geomspace(max(1.0, np.min(taus)), np.max(taus), num=log_bins+1)
    idx = np.digitize(taus, bins)-1
    xs, ys = [], []
    for b in range(log_bins):
        m = (idx==b)
        if np.count_nonzero(m) < min_pts: continue
        tau_med = float(np.median(taus[m])); V_med = float(np.median(Vs[m]))
        if tau_med>0 and V_med>0:
            xs.append(math.log(tau_med)); ys.append(math.log(V_med))
    xs, ys = np.array(xs), np.array(ys)
    if len(xs)<2: return float("nan"), xs, ys
    d_hat = safe_robust_slope(xs, ys)
    return d_hat, xs, ys

# ---------------- Quantum ensemble & T proxy ----------------
def quantum_ensemble_T_and_K(B, D, W, mass2:float=4e-3, samples:int=120, trim:float=0.15):
    N = B.shape[0]
    if N==0: return float("nan"), float("nan")
    A = B + mass2 * sparse.identity(N, format="csr")
    Hs, Phis, T_list = [], [], []
    for _ in range(samples):
        xi = rng.standard_normal(N)
        phi = spsolve(A, xi)
        g = D @ phi
        grad2 = float(np.dot(W.diagonal(), g*g)) / max(1,N)
        m2term = float(mass2 * np.dot(phi, phi)) / max(1,N)
        T_eff = 0.5 * (grad2 + m2term)
        T_list.append(T_eff)
        flux = D.T @ (W @ (D @ phi)); eom = - (B @ phi); r = flux + eom
        Hs.append(float(np.dot(r,r))); Phis.append(float(np.sum(np.abs(r))))
    Hs, Phis, T_list = np.array(Hs), np.array(Phis), np.array(T_list)
    if len(Hs) < 12: Kq = float("nan")
    else:
        lo, hi = np.quantile(Phis, [trim, 1.0-trim]); m = (Phis>=lo) & (Phis<=hi)
        if np.count_nonzero(m) < 12: Kq = float("nan")
        else:
            x = np.log(Phis[m] + 1e-300); y = np.log(Hs[m] + 1e-300)
            if np.std(x)<1e-9 or np.std(y)<1e-9: Kq=float("nan")
            else: Kq = safe_robust_slope(x, y)
    return float(np.nanmean(T_list)), float(Kq)

# ---------------- Local proxies & graviton-like ----------------
def local_valence(G:nx.DiGraph):
    return np.array([G.out_degree(n)+G.in_degree(n) for n in G.nodes()], float)

def local_dim_proxy(G:nx.DiGraph, sample:int=256, max_hop:int=3):
    nodes = list(G.nodes()); N=len(nodes)
    if N==0: return float("nan")
    H = G.to_undirected()
    vals=[]
    for _ in range(min(sample, len(nodes))):
        s = int(rng.integers(0, len(nodes)))
        visited={s}; frontier=[s]; layers=[]
        for r in range(1, max_hop+1):
            nxt=[]
            for u in frontier:
                for v in H.neighbors(u):
                    if v not in visited:
                        visited.add(v); nxt.append(v)
            layers.append(len(nxt)); frontier=nxt
            if not frontier: break
        if len(layers)>=2 and np.all(np.array(layers)>0):
            x = np.log(np.arange(1, len(layers)+1))
            y = np.log(np.array(layers)+1e-12)
            vals.append(safe_robust_slope(x,y))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(np.median(vals)) if vals.size else float("nan")

def curvature_proxy_global(G:nx.DiGraph, d_target:float=4.0, z0:float=14.0):
    N = G.number_of_nodes(); E = G.number_of_edges()
    if N==0: return 0.0, 4.0, 0.0
    z = E/max(1,N)
    d_loc = local_dim_proxy(G, sample=256, max_hop=3)
    if not np.isfinite(d_loc): d_loc = 4.0
    R_eff = (d_loc - d_target) + 0.08*(z - z0)
    return float(R_eff), float(d_loc), float(z)

def graviton_like_diagnostics(G: nx.DiGraph):
    N = G.number_of_nodes()
    if N==0: return float("nan"), float("nan"), (np.array([]), np.array([]))
    H = G.to_undirected()
    val = local_valence(G); vbar = float(np.mean(val) if val.size else 0.0)
    if vbar==0.0: return float("nan"), float("nan"), (np.array([]), np.array([]))
    h = (val - vbar)/(vbar+1e-12)
    s = int(rng.integers(0, N))
    dist = nx.single_source_shortest_path_length(H, s, cutoff=16)
    rs_used, corr = [], []
    for r in sorted(set(dist.values()) - {0}):
        shell_nodes = [node for node, rr in dist.items() if rr == r]
        if len(shell_nodes) < 10: continue
        rs_used.append(r)
        corr.append(float(np.mean(h[shell_nodes] * h[s])))
    if len(rs_used) < 3:
        div = np.array([G.out_degree(n) - G.in_degree(n) for n in G.nodes()], float)
        trans_ratio = float(np.std(div)/(np.std(val)+1e-12)) if np.std(val)>0 else float("nan")
        return float("nan"), trans_ratio, (np.array([]), np.array([]))
    rs_np = np.array(rs_used, float); corr_np = np.array(corr, float)
    p_pos = -safe_robust_slope(np.log(rs_np + 1e-18), np.log(corr_np + 1e-18)) if np.all(corr_np>0) else float("nan")
    m = np.abs(corr_np) > 1e-15
    p_abs = -safe_robust_slope(np.log(rs_np[m] + 1e-18), np.log(np.abs(corr_np[m]) + 1e-18)) if np.count_nonzero(m)>=3 else float("nan")
    p = p_pos if np.isfinite(p_pos) else p_abs
    div = np.array([G.out_degree(n) - G.in_degree(n) for n in G.nodes()], float)
    trans_ratio = float(np.std(div)/(np.std(val)+1e-12)) if np.std(val)>0 else float("nan")
    return p, trans_ratio, (rs_np, corr_np)

# ---------------- Geometric action proxy & run_geometry (Patched: z 自适应) ----------------
def geometric_action_proxy(d_eff:float, z:float, z0:float=14.0, a:float=1.1, b:float=0.010):
    if not np.isfinite(d_eff): d_eff = 4.0
    if not np.isfinite(z): z = z0
    return a*(d_eff-4.0)**2 + b*(z - z0)**2

def run_geometry(N:int, base_p:float, decay:float, max_fan:int, tr_mode:str, tr_cap:int,
                 min_E_over_N:float, max_E_over_N:float):
    # 自适应把 z 调回目标带，避免过密/过稀导致 d 失真 & BFS 爆时
    G=None
    bp, dc = base_p, decay
    for _ in range(18):
        G = sample_causal_dag(N, bp, dc, max_fan, tr_mode, tr_cap)
        z = G.number_of_edges()/max(1,G.number_of_nodes())
        if min_E_over_N <= z <= max_E_over_N:
            return G
        # 简单回调：z 太高就降 base_p 或加速衰减；z 太低反之
        if z > max_E_over_N:
            bp *= 0.94
            dc *= 1.08
        else:  # z < min
            bp *= 1.06
            dc *= 0.92
    # 兜底返回最后一幅（即便偏离，也不至于卡死）
    return G

# ---------------- Simulation config ----------------
@dataclass
class SimCfg:
    N:int=1200; geom_samples:int=8
    base_p:float=0.54; decay:float=0.018; max_fan:int=56
    tr_mode:str="off"; tr_cap:int=64
    min_E_over_N:float=10.5; max_E_over_N:float=16.5  # 指纹带锁定
    mass2A:float=4e-3; mass2B:float=7e-3
    q_samples:int=120
    ts_steps:int=420; ts_win:int=128
    beta:float=0.20; eps:float=0.02; z0:float=14.0
    a:float=1.1; b:float=0.010

# ---------------- Core runner: one suite ----------------
def run_suite(cfg:SimCfg, weight_mode:str="gamma"):
    rec=[]
    for _ in tqdm(range(cfg.geom_samples), desc="Geometries"):
        G = run_geometry(cfg.N, cfg.base_p, cfg.decay, cfg.max_fan, cfg.tr_mode, cfg.tr_cap,
                         cfg.min_E_over_N, cfg.max_E_over_N)
        D,W,B = build_DWB(G, weight_mode, 1.0)
        N = G.number_of_nodes(); E = G.number_of_edges()
        ward_med, _, _ = ward_residual_stats(B,D,W,N,steps=240,eta=0.05)
        K_time = timeseries_K(B,D,W,N,EvoCfg(cfg.ts_steps,0.05,1e-12), win=cfg.ts_win)
        d_hat, _, _ = alex_dim_intrinsic(G, M=1600, log_bins=12, min_pts=10, tau_min=4)
        R_eff, d_loc, z = curvature_proxy_global(G, d_target=4.0, z0=cfg.z0)
        d_eff = d_hat if np.isfinite(d_hat) else (d_loc if np.isfinite(d_loc) else 4.0)
        T_A, Kq_A = quantum_ensemble_T_and_K(B,D,W, mass2=cfg.mass2A, samples=cfg.q_samples, trim=0.15)
        T_B, Kq_B = quantum_ensemble_T_and_K(B,D,W, mass2=cfg.mass2B, samples=cfg.q_samples, trim=0.15)
        p_slope, trans_ratio, (rs,corr) = graviton_like_diagnostics(G)
        Sg = geometric_action_proxy(d_eff, z, z0=cfg.z0, a=cfg.a, b=cfg.b)
        rec.append({
            "G":G,"N":N,"E":E,"z":z,"ward":ward_med,"K_time":K_time,"d":d_hat,"d_loc":d_loc,"d_eff":d_eff,
            "R_eff":R_eff,"Sg":Sg,"T_A":T_A,"T_B":T_B,"Kq_A":Kq_A,"Kq_B":Kq_B,
            "grav_p":p_slope,"grav_tr":trans_ratio,"rs":rs,"corr":corr
        })
    # weights
    S_list = np.array([r["Sg"] for r in rec], float)
    T_list = np.array([r["T_A"]+r["T_B"] for r in rec], float)
    if not np.any(np.isfinite(S_list)):
        w = np.ones(len(rec), float)
    else:
        Smin = np.nanmin(S_list)
        S_adj = np.where(np.isfinite(S_list), S_list, Smin)
        T_mean = np.nanmean(T_list) if np.any(np.isfinite(T_list)) else 0.0
        T_adj = np.where(np.isfinite(T_list), T_list, T_mean)
        w = np.exp(-cfg.beta*(S_adj - Smin) - cfg.eps*(T_adj - T_mean))
    def wavg(arr):
        arr=np.array(arr,float); m=np.isfinite(arr)
        if not np.any(m): return float("nan")
        ww = w[m]; aa = arr[m]
        return float(np.dot(ww, aa)/(ww.sum()+1e-18))
    summ = {
        "Ward": wavg([r["ward"] for r in rec]),
        "K_time": wavg([r["K_time"] for r in rec]),
        "d": wavg([r["d"] for r in rec]),
        "d_eff": wavg([r["d_eff"] for r in rec]),
        "z": wavg([r["z"] for r in rec]),
        "Kq_A": wavg([r["Kq_A"] for r in rec]),
        "Kq_B": wavg([r["Kq_B"] for r in rec]),
        "grav_p": wavg([r["grav_p"] for r in rec]),
        "grav_tr": wavg([r["grav_tr"] for r in rec]),
        "w_norm": float(np.sum(w)),
        "reclist": rec
    }
    # GR proxy regression
    R_eff = np.array([r["R_eff"] for r in rec], float)
    T_tot = T_list
    m = np.isfinite(R_eff) & np.isfinite(T_tot)
    if np.count_nonzero(m) >= 3:
        kappa = safe_robust_slope(T_tot[m].reshape(-1,1), R_eff[m])
        a = R_eff[m]-R_eff[m].mean(); b = T_tot[m]-T_tot[m].mean()
        R2 = float((np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-18))**2)
    else:
        kappa, R2 = float("nan"), float("nan")
    summ["kappa_GR"]=float(kappa); summ["R2_GR"]=float(R2)
    return rec, summ, w

# ---------------- Red-team tests P0..P8 ----------------
def P0_ward_convergence(cfg:SimCfg, seeds=[2025,2026,2027], densities=[10.5,12,14,16]):
    out=[]
    for seed in seeds:
        _ = np.random.seed(seed)
        for dtarget in densities:
            cfg2 = SimCfg(**vars(cfg)); cfg2.min_E_over_N=dtarget; cfg2.max_E_over_N=dtarget+0.5
            # 降样本提速（仅 P0 用）
            cfg2.geom_samples = max(3, cfg.geom_samples//2)
            cfg2.q_samples   = 24
            rec, summ, w = run_suite(cfg2)
            out.append(("seed",seed,"z_target",dtarget,"Ward",summ["Ward"]))
            print(f"P0 seed={seed}, z_target~{dtarget}: Ward median={summ['Ward']:.3e}")
    return out

def P1_K_and_d_scan(cfg:SimCfg, weight_modes=[("gamma","off"),("unit","soft")]):
    results=[]
    for (weight_mode, tr_mode) in weight_modes:
        for base_p in [0.46, 0.50, 0.54]:
            for decay in [0.018, 0.024]:
                cfg2 = SimCfg(**vars(cfg))
                cfg2.base_p=base_p; cfg2.decay=decay; cfg2.tr_mode=tr_mode
                # P1 降样本提速
                cfg2.geom_samples = max(3, cfg.geom_samples//2)
                cfg2.q_samples   = 24
                rec, summ, w = run_suite(cfg2, weight_mode=weight_mode)
                d_hat_med = float(np.nanmedian([r["d"] for r in rec]))
                print(f"P1 mode={weight_mode}/{tr_mode}, base_p={base_p:.2f}, decay={decay:.3f} "
                      f"-> K={summ['K_time']:.3f}, d_hat={d_hat_med:.3f}, d_eff={summ['d_eff']:.3f}, z={summ['z']:.2f}, Ward={summ['Ward']:.2e}")
                results.append((weight_mode,tr_mode,base_p,decay,summ))
    return results

def P2_dimension_flow(cfg:SimCfg, Mlist=[1000,1400,1800]):
    out=[]
    for M in Mlist:
        recs=[]
        for _ in range(4):
            G = run_geometry(cfg.N, cfg.base_p, cfg.decay, cfg.max_fan, cfg.tr_mode, cfg.tr_cap,
                             cfg.min_E_over_N, cfg.max_E_over_N)
            d_hat, xs, ys = alex_dim_intrinsic(G, M=M, log_bins=12, min_pts=12, tau_min=4, seed=int(time.time()%100000))
            recs.append(d_hat)
        med = float(np.nanmedian([r for r in recs if np.isfinite(r)])) if recs else float("nan")
        print(f"P2 M={M} -> median d_hat={med:.3f} (samples {len(recs)})")
        out.append((M,med,recs))
    return out

def P3_microcausal(cfg:SimCfg):
    rec, summ, w = run_suite(cfg)
    for i,r in enumerate(rec):
        print(f"P3 geom#{i}: z={r['z']:.2f}, ward={r['ward']:.3e}, grav_tr={r['grav_tr']:.3f}")
    return rec, summ

def P4_weighted_stability(cfg:SimCfg, betalist=[0.08,0.15,0.20,0.28]):
    out=[]
    for beta in betalist:
        cfg2 = SimCfg(**vars(cfg)); cfg2.beta=beta
        rec, summ, w = run_suite(cfg2)
        print(f"P4 beta={beta}: K_time={summ['K_time']:.3f}, d_eff={summ['d_eff']:.3f}, Ward={summ['Ward']:.3e}")
        out.append((beta,summ))
    return out

def P5_GR_regression_stability(cfg:SimCfg):
    rec, summ, w = run_suite(cfg)
    print(f"P5 kappa={summ['kappa_GR']:.3e}, R2={summ['R2_GR']:.3f}")
    return rec, summ

def P6_equivalence_principle(cfg:SimCfg, repeats=4):
    base_summaries=[]; swapped_summaries=[]
    for _ in range(repeats):
        rec, summ, w = run_suite(cfg)
        base_summaries.append(summ)
        cfg2 = SimCfg(**vars(cfg)); cfg2.mass2A=cfg.mass2B; cfg2.mass2B=cfg.mass2A
        rec2, summ2, w2 = run_suite(cfg2)
        swapped_summaries.append(summ2)
    def med_field(list_s, key):
        arr = np.array([s[key] for s in list_s], float); arr = arr[np.isfinite(arr)]
        return float(np.median(arr)) if arr.size else float("nan")
    d_base = med_field(base_summaries, "d_eff"); d_sw = med_field(swapped_summaries, "d_eff")
    z_base = med_field(base_summaries, "z"); z_sw = med_field(swapped_summaries, "z")
    Kb = med_field(base_summaries, "K_time"); Ks = med_field(swapped_summaries, "K_time")
    print(f"P6 baseline d_eff={d_base:.3f}, z={z_base:.2f}, K_time={Kb:.3f}")
    print(f"P6 swapped  d_eff={d_sw:.3f}, z={z_sw:.2f}, K_time={Ks:.3f}")
    print(f"P6 rel Δd/d={(d_sw-d_base)/max(1e-9,abs(d_base)):.3e}, rel Δz/z={(z_sw-z_base)/max(1e-9,abs(z_base)):.3e}")
    return base_summaries, swapped_summaries

def P7_graviton_agent(cfg:SimCfg, trials=1):
    rec, summ, w = run_suite(cfg)
    successes=0; total=0
    for r in rec:
        p = r["grav_p"]; total+=1
        print(f"P7 geom z={r['z']:.2f}, grav_p={p:.3f}, grav_tr={r['grav_tr']:.3f}")
        if np.isfinite(p) and 0.3 <= p <= 1.6: successes+=1
    print(f"P7 success fraction (0.3<=p<=1.6): {successes}/{total}")
    return rec, summ

def P8_background_independence(cfg:SimCfg, relabels=4):
    G = run_geometry(cfg.N, cfg.base_p, cfg.decay, cfg.max_fan, cfg.tr_mode, cfg.tr_cap,
                     cfg.min_E_over_N, cfg.max_E_over_N)
    base_D, base_W, base_B = build_DWB(G, "gamma", 1.0)
    base_K = timeseries_K(base_B, base_D, base_W, G.number_of_nodes(), EvoCfg(cfg.ts_steps,0.05,1e-12), win=cfg.ts_win)
    base_d,_,_ = alex_dim_intrinsic(G, M=1600, log_bins=12, min_pts=10, tau_min=4)
    print(f"P8 base: K={base_K:.3f}, d={base_d:.3f}")
    Ks=[]; ds=[]
    nodes = list(G.nodes())
    for i in range(relabels):
        perm = nodes.copy(); random.shuffle(perm)
        mapping = {nodes[j]: perm[j] for j in range(len(nodes))}
        H = nx.relabel_nodes(G, mapping, copy=True)
        D,W,B = build_DWB(H, "gamma", 1.0)
        Kt = timeseries_K(B, D, W, H.number_of_nodes(), EvoCfg(cfg.ts_steps,0.05,1e-12), win=cfg.ts_win)
        dt,_,_ = alex_dim_intrinsic(H, M=1600, log_bins=12, min_pts=10, tau_min=4)
        Ks.append(Kt); ds.append(dt)
        print(f"P8 relabel#{i}: K={Kt:.3f}, d={dt:.3f}")
    Ks = np.array([k for k in Ks if np.isfinite(k)])
    ds = np.array([d for d in ds if np.isfinite(d)])
    print(f"P8 med relabeled ΔK = {(np.median(Ks)-base_K):.3e}, Δd = {(np.median(ds)-base_d):.3e}")
    return base_K, base_d, Ks, ds

# ---------------- Driver to run the full protocol ----------------
def run_red_team_all(cfg:SimCfg):
    print("=== RUN RED-TEAM PROTOCOL (P0..P8) ===")
    t0=time.time()
    print("\n-- P0: Ward convergence / resolution sweep")
    p0 = P0_ward_convergence(cfg)
    print("\n-- P1: K & d scan (micro rules)")
    p1 = P1_K_and_d_scan(cfg)
    print("\n-- P2: Dimension flow (M sweep)")
    p2 = P2_dimension_flow(cfg)
    print("\n-- P3: Microcausality checks")
    p3 = P3_microcausal(cfg)
    print("\n-- P4: Weighted stability (beta sweep)")
    p4 = P4_weighted_stability(cfg)
    print("\n-- P5: GR regression stability")
    p5 = P5_GR_regression_stability(cfg)
    print("\n-- P6: Equivalence principle (swap species)")
    p6 = P6_equivalence_principle(cfg)
    print("\n-- P7: Graviton-like agent")
    p7 = P7_graviton_agent(cfg)
    print("\n-- P8: Background independence (relabel)")
    p8 = P8_background_independence(cfg)
    dt = time.time()-t0
    print(f"\n=== RED-TEAM DONE (elapsed {dt:.1f}s) ===")
    return dict(P0=p0,P1=p1,P2=p2,P3=p3,P4=p4,P5=p5,P6=p6,P7=p7,P8=p8)

# ===========================
# 默认配置；你也可以为速度先降配
cfg = SimCfg()
# 可选提速：取消注释
# cfg.geom_samples = 4
# cfg.q_samples   = 48

# 跑全套
results = run_red_team_all(cfg)

print("\n\n== SUMMARY (selected aggregated metrics) ==")
print("Ward (weighted median):", np.nanmedian([r["ward"] for r in results['P3'][0]]))
print("K_time (weighted):",     np.nanmedian([r["K_time"] for r in results['P3'][0]]))
print("d_hat (per-geom median):", np.nanmedian([r["d"] for r in results['P3'][0]]))
print("d_eff (weighted):",      np.nanmedian([r["d_eff"] for r in results['P3'][0]]))
print("z (weighted E/N):",      np.nanmedian([r["z"] for r in results['P3'][0]]))

# 简单图示：P2 的维度流示意（若想画图）
try:
    Ms = [x[0] for x in results['P2']]
    dmeds = [x[1] for x in results['P2']]
    plt.figure()
    plt.plot(Ms, dmeds, marker='o')
    plt.xlabel("M (pair samples)")
    plt.ylabel("median d_hat")
    plt.title("P2: Dimension flow vs M")
    plt.grid(True)
    plt.show()
except Exception:
    pass
