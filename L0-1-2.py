# =========================================================
# Main verification script for Appendix B:
#   (1) L1 → L2 convergence: -D^T J / h → dJ/dx
#   (2) L0 red-team: B(ε) = L + ε R, Ward(ε) vs div(ε)
# =========================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

# ---------- Common utilities ----------

def incidence_matrix_oriented(G):
    """
    Return oriented incidence matrix D of shape (|E| x |V|).
    networkx.incidence_matrix returns (|V| x |E|), so we transpose.
    """
    M = nx.incidence_matrix(G, oriented=True)
    if hasattr(M, "toarray"):
        return M.toarray().T
    else:
        return np.array(M).T

def laplacian_from_D(D, w_edge=None):
    """
    Build L = D^T W_E D from incidence matrix D (|E| x |V|).
    """
    m = D.shape[0]
    if w_edge is None:
        W_E = np.eye(m)
    else:
        W_E = np.diag(w_edge)
    L = D.T @ W_E @ D
    return L, W_E

# =========================================================
# Part 1: L1 → L2 convergence
# =========================================================

def line_graph(N):
    """1D line graph with nodes 0..N-1 and edges (i, i+1)."""
    return nx.path_graph(N)

def test_divergence_convergence(N_list):
    errors = []
    hs = []
    for N in N_list:
        G = line_graph(N)
        D = incidence_matrix_oriented(G)      # (E x N)
        L_lin, W_E_lin = laplacian_from_D(D)

        h = 1.0 / (N - 1)
        hs.append(h)

        # node and edge positions
        x_nodes = np.linspace(0, 1, N)
        edge_centers = []
        for (i, j) in G.edges():
            edge_centers.append(0.5 * (x_nodes[i] + x_nodes[j]))
        edge_centers = np.array(edge_centers)

        # continuous J and its derivative
        J_edge   = np.sin(2 * np.pi * edge_centers)
        div_true = 2 * np.pi * np.cos(2 * np.pi * x_nodes)

        # discrete divergence: -D^T J / h
        div_disc = -(D.T @ J_edge) / h

        err = np.linalg.norm(div_disc - div_true, ord=2) / np.sqrt(N)
        errors.append(err)

    return np.array(hs), np.array(errors)

# run L1→L2 test
N_list = [10, 20, 40, 80, 160, 320]
hs, errors = test_divergence_convergence(N_list)

plt.figure(figsize=(6,4))
plt.loglog(hs, errors, 'o-', label=r"$\|\mathrm{div}_h J - \partial_x J\|_2$")
plt.loglog(hs, hs, '--', label=r"$\mathcal{O}(h)$ reference")
plt.gca().invert_xaxis()
plt.xlabel(r"$h = 1/(N-1)$")
plt.ylabel("L2 error")
plt.title(r"L1 $\rightarrow$ L2: $-D^\top J/h \to \partial_x J$")
plt.grid(True, which="both")
plt.legend()
plt.show()

print("=== L1 → L2 收敛实验结果 ===")
print("h, error:")
for h, e in zip(hs, errors):
    print(f"h={h:.4f}, error={e:.4e}")
print()

# =========================================================
# Part 2: L0 red-team: B(ε) = L + ε R
# =========================================================

def make_ring_graph(n_nodes: int):
    """Ring graph with |E| = |V|."""
    return nx.cycle_graph(n_nodes)

# 1. build closed L0 operator
n_nodes = 40
G_ring = make_ring_graph(n_nodes)
D_ring = incidence_matrix_oriented(G_ring)   # (E x N)
L_ring, W_E_ring = laplacian_from_D(D_ring)

print("Graph info:")
print("  |V| =", n_nodes)
print("  |E| =", D_ring.shape[0])
print("  L shape:", L_ring.shape)
print()

# random symmetric defect R
np.random.seed(0)
R = np.random.randn(*L_ring.shape)
R = 0.5 * (R + R.T)

def evolve_phi(B, phi0, n_steps=300, dt=0.05):
    """
    Simple diffusion-like iteration:
        phi_{k+1} = phi_k - dt * B phi_k
    Returns phi(k) over time.
    """
    n = len(phi0)
    phi_hist = np.zeros((n_steps, n))
    phi = phi0.copy()
    for k in range(n_steps):
        phi_hist[k] = phi
        phi = phi - dt * (B @ phi)
    return phi_hist

def ward_stat(L, B, phi_hist):
    """
    Ward(ε) ≈ median_k ||(L-B)phi_k|| / ||B phi_k||.
    """
    chi_norm = []
    ref_norm = []
    for phi in phi_hist:
        chi = (L - B) @ phi
        Bphi = B @ phi
        chi_norm.append(np.linalg.norm(chi))
        ref_norm.append(np.linalg.norm(Bphi) + 1e-15)
    chi_norm = np.array(chi_norm)
    ref_norm = np.array(ref_norm)
    ratio = chi_norm / ref_norm
    return np.median(ratio)

def div_norm(D, W_E, phi_hist):
    """
    Discrete divergence norm:
        J_k = W_E D phi_k
        div_k = D^T J_k
    Return median_k ||div_k||_2.
    """
    norms = []
    for phi in phi_hist:
        J = W_E @ (D @ phi)
        div = D.T @ J
        norms.append(np.linalg.norm(div))
    return np.median(norms)

# 2. scan ε and measure Ward(ε), div(ε)
eps_list = np.logspace(-6, -1, 10)
n_steps = 300
dt = 0.05
n_trials = 8

ward_vals = []
div_vals  = []

print("=== L0 破坏红队：eps, Ward, div ===")
for eps in eps_list:
    B = L_ring + eps * R
    this_wards = []
    this_divs  = []
    for seed in range(n_trials):
        np.random.seed(seed)
        phi0 = np.random.randn(n_nodes)
        phi_hist = evolve_phi(B, phi0, n_steps=n_steps, dt=dt)
        this_wards.append(ward_stat(L_ring, B, phi_hist))
        this_divs.append(div_norm(D_ring, W_E_ring, phi_hist))
    w_med = np.median(this_wards)
    d_med = np.median(this_divs)
    ward_vals.append(w_med)
    div_vals.append(d_med)
    print(f"eps = {eps:.1e},  Ward ~ {w_med:.3e},  div ~ {d_med:.3e}")

ward_vals = np.array(ward_vals)
div_vals  = np.array(div_vals)

# 3. visualization
fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].loglog(eps_list, ward_vals, 'o-', label="Ward(ε)")
ax[0].loglog(eps_list, div_vals,  's--', label="div(ε)")
ax[0].set_xlabel(r"$\varepsilon$ (strength of $L_0$ defect)")
ax[0].set_ylabel("median value over time")
ax[0].set_title("Closure defect vs Ward & divergence")
ax[0].grid(True, which="both")
ax[0].legend()

ax[1].loglog(ward_vals, div_vals, 'o-')
ax[1].set_xlabel("Ward(ε)")
ax[1].set_ylabel("median ||D^T J||")
ax[1].set_title("Ward vs divergence")
ax[1].grid(True, which="both")

plt.tight_layout()
plt.show()
