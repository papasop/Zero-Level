# @title Spectral dimension of prime-difference graphs (full test)
import numpy as np
from sympy import primerange
import random
import matplotlib.pyplot as plt

# ------------------------------
# 0. global params
# ------------------------------
M = 600                                # number of primes
K_list = [10,20,30,40,50,60,70,80,100,120]
ts = np.logspace(-3, 2, 90)            # t-grid for heat trace
max_eigs = 650                         # how many eigenvalues to use

# ------------------------------
# 1. get primes
# ------------------------------
primes = list(primerange(2, 1000000))[:M]

# ------------------------------
# 2. helpers
# ------------------------------
def build_prime_diff_L(M, primes, H, beta=0.0):
    """
    Build weighted Laplacian for prime-difference graph:
      vertices: first M primes
      edges: (i,j) if p_j - p_i in H, i<j
      weight: w_ij = (log p_i log p_j)^(-(1+beta))
    return: L (MxM), edges
    """
    P = primes
    edges = []
    for i, pi in enumerate(P):
        for j in range(i+1, M):
            pj = P[j]
            diff = pj - pi
            if diff in H:
                edges.append((i, j))

    E = len(edges)
    D = np.zeros((E, M), dtype=float)
    W = np.zeros((E, E), dtype=float)
    for k, (i, j) in enumerate(edges):
        D[k, i] = -1.0
        D[k, j] =  1.0
        w = 1.0 / ((np.log(P[i]) * np.log(P[j])) ** (1.0 + beta))
        W[k, k] = w

    L = D.T @ W @ D
    L = 0.5*(L + L.T)
    return L, edges

def spectral_dimension(L, ts, kmax=650):
    """
    P(t) = (1/N) sum exp(-t lambda)
    d_s(t) = -2 d/dlogt log P(t)
    d_s   = max_t d_s(t)
    """
    vals, _ = np.linalg.eigh(L)
    vals = np.maximum(vals, 0.0)
    vals = vals[:min(kmax, len(vals))]
    N = L.shape[0]

    Pts = []
    for t in ts:
        Pts.append(np.exp(-t * vals).sum() / N)
    Pts = np.array(Pts)

    logt = np.log(ts)
    logP = np.log(Pts + 1e-300)
    dlogP_dlogt = np.gradient(logP, logt)
    d_eff = -2.0 * dlogP_dlogt
    d_s = float(d_eff.max())
    return d_s, d_eff

def spectral_gap(L):
    vals, _ = np.linalg.eigh(L)
    vals = np.maximum(vals, 0.0)
    for v in vals:
        if v > 1e-10:
            return v
    return 0.0

# ------------------------------
# 3. main: closed graphs
# ------------------------------
closed_results = []

print("=== (1) closed prime-difference graphs ===")
for K in K_list:
    H = set(range(1, K+1))
    L, edges = build_prime_diff_L(M, primes, H)
    d_s, d_eff = spectral_dimension(L, ts, kmax=max_eigs)
    lam2 = spectral_gap(L)
    avg_deg = 2*len(edges)/M
    closed_results.append((K, len(edges), avg_deg, d_s, lam2))
    print(f"K={K:3d} |E|={len(edges):5d}  avg_deg={avg_deg:6.2f}  d_s={d_s:6.3f}  lambda_2={lam2:.3e}")

# ------------------------------
# 4. broken-closure test: remove 5% edges
# ------------------------------
broken_results = []

print("\n=== (2) broken-closure graphs (remove 5% edges) ===")
for K in K_list:
    H = set(range(1, K+1))
    L, edges = build_prime_diff_L(M, primes, H)
    E = len(edges)
    # remove 5% edges randomly
    n_remove = max(1, int(0.05 * E))
    remove_idx = set(random.sample(range(E), n_remove))
    edges2 = [e for i, e in enumerate(edges) if i not in remove_idx]

    # rebuild L from edges2
    D2 = np.zeros((len(edges2), M))
    W2 = np.zeros((len(edges2), len(edges2)))
    for k, (i, j) in enumerate(edges2):
        D2[k, i] = -1.0
        D2[k, j] =  1.0
        w = 1.0 / ((np.log(primes[i]) * np.log(primes[j])) ** 1.0)
        W2[k, k] = w
    L2 = D2.T @ W2 @ D2
    L2 = 0.5*(L2 + L2.T)

    d_s2, _ = spectral_dimension(L2, ts, kmax=max_eigs)
    lam2_2 = spectral_gap(L2)
    avg_deg2 = 2*len(edges2)/M
    broken_results.append((K, len(edges2), avg_deg2, d_s2, lam2_2))
    print(f"[broken] K={K:3d} |E|={len(edges2):5d}  avg_deg={avg_deg2:6.2f}  d_s={d_s2:6.3f}  lambda_2={lam2_2:.3e}")

# ------------------------------
# 5. random-H test: same |H|, random diffs
# ------------------------------
random_results = []

print("\n=== (3) random-difference graphs (same |H|) ===")
max_diff = primes[-1] - primes[0]   # largest possible diff in prime set
for K in K_list:
    # sample K random diffs from [1, max_diff]
    H_rand = set(random.sample(range(1, max_diff+1), K))
    Lr, edges_r = build_prime_diff_L(M, primes, H_rand)
    d_sr, _ = spectral_dimension(Lr, ts, kmax=max_eigs)
    lam2_r = spectral_gap(Lr)
    avg_deg_r = 2*len(edges_r)/M
    random_results.append((K, len(edges_r), avg_deg_r, d_sr, lam2_r))
    print(f"[random] K={K:3d} |E|={len(edges_r):5d}  avg_deg={avg_deg_r:6.2f}  d_s={d_sr:6.3f}  lambda_2={lam2_r:.3e}")

# ------------------------------
# 6. (optional) quick plots
# ------------------------------
plt.figure(figsize=(6,4))
plt.plot([r[0] for r in closed_results], [r[3] for r in closed_results], "o-", label="closed")
plt.plot([r[0] for r in broken_results], [r[3] for r in broken_results], "s--", label="broken (5%)")
plt.plot([r[0] for r in random_results], [r[3] for r in random_results], "x-", label="random-H")
plt.axhline(4.0, color="r", linestyle="--", label="4D collar")
plt.xlabel("K in H={1..K}")
plt.ylabel("spectral dimension d_s")
plt.title("d_s vs K (prime-difference graphs)")
plt.grid(True)
plt.legend()
plt.show()


