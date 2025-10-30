# @title Collar-to-4 with controls (closed vs edge-removed vs random-H)
import numpy as np
from sympy import primerange
import matplotlib.pyplot as plt

# ---------- common builders ----------
def build_prime_diff_graph(M, H, beta=0.0):
    primes = list(primerange(2, 1000000))
    P = primes[:M]
    edges = []
    for i, pi in enumerate(P):
        for j in range(i+1, M):
            pj = P[j]
            if (pj - pi) in H:
                edges.append((i, j))
    E = len(edges)
    N = M
    D = np.zeros((E, N), dtype=float)
    W = np.zeros((E, E), dtype=float)
    for k, (i, j) in enumerate(edges):
        D[k, i] = -1.0
        D[k, j] =  1.0
        w = 1.0 / ((np.log(P[i]) * np.log(P[j]))**(1.0 + beta))
        W[k, k] = w
    L = D.T @ W @ D
    L = 0.5 * (L + L.T)
    return L, edges, P

def spectral_dimension(L, ts, k=650):
    vals, _ = np.linalg.eigh(L)
    vals = np.maximum(vals, 0.0)
    vals_k = vals[:min(k, len(vals))]
    N = L.shape[0]
    P = []
    for t in ts:
        P.append(np.exp(-t * vals_k).sum() / N)
    P = np.array(P)
    logt = np.log(ts)
    logP = np.log(P + 1e-300)
    dlogP_dlogt = np.gradient(logP, logt)
    d_eff = -2.0 * dlogP_dlogt
    return d_eff

M = 600
K_list = [10,20,30,40,50,60,70,80]
ts = np.logspace(-3, 2, 90)

closed_dmax = []
broken_dmax = []
random_dmax = []

print("=== closed collar ===")
stored_edges = {}
stored_primes = {}

for K in K_list:
    H = tuple(range(1, K+1))
    L, edges, Pnodes = build_prime_diff_graph(M, H)
    stored_edges[K] = edges
    stored_primes[K] = Pnodes
    d_eff = spectral_dimension(L, ts, k=650)
    dmax = float(d_eff.max())
    closed_dmax.append(dmax)
    print(f"H=1..{K:2d} -> d_max(closed) = {dmax:.3f}")

print("\n=== broken-closure collar (remove 5% edges) ===")
for K in K_list:
    edges = stored_edges[K]
    Pnodes = stored_primes[K]
    E = len(edges)
    cut = max(1, int(0.05 * E))
    edges_broken = edges[cut:]  # 删前 5%
    # 重建 L
    N = M
    D2 = np.zeros((E-cut, N))
    W2 = np.zeros((E-cut, E-cut))
    for idx, (i, j) in enumerate(edges_broken):
        D2[idx, i] = -1.0
        D2[idx, j] =  1.0
        w = 1.0 / ((np.log(Pnodes[i]) * np.log(Pnodes[j]))**1.0)
        W2[idx, idx] = w
    L2 = D2.T @ W2 @ D2
    L2 = 0.5*(L2 + L2.T)
    d_eff2 = spectral_dimension(L2, ts, k=650)
    dmax2 = float(d_eff2.max())
    broken_dmax.append(dmax2)
    print(f"H=1..{K:2d} -> d_max(broken) = {dmax2:.3f}")

print("\n=== random-H collar (same |H|, random diffs) ===")
max_diff = 200  # 随机挑的差的上界
rng = np.random.default_rng(0)
for K in K_list:
    # 随机选 K 个差
    Hrand = tuple(sorted(rng.choice(np.arange(1, max_diff+1), size=K, replace=False).tolist()))
    Lr, _, _ = build_prime_diff_graph(M, Hrand)
    d_eff_r = spectral_dimension(Lr, ts, k=650)
    dmax_r = float(d_eff_r.max())
    random_dmax.append(dmax_r)
    print(f"|H|={K:2d} (random) -> d_max = {dmax_r:.3f}")

# 画图
plt.figure(figsize=(7,5))
plt.plot(K_list, closed_dmax, "o-", label="closed, H=1..K")
plt.plot(K_list, broken_dmax, "s--", label="5% edges removed")
plt.plot(K_list, random_dmax, "x:", label="random diffs, |H|=K")
plt.axhline(4.0, color="r", linestyle="--", label="d=4 collar")
plt.xlabel("K in H=1..K (or |H|)")
plt.ylabel("max spectral dimension d_max")
plt.ylim(0,5)
plt.title("Collar-to-4 with controls (M=600)")
plt.grid(True)
plt.legend()
plt.show()

