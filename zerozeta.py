# ================================================================
# Native DFT of Primes — Colab Full V6 (final + robust + nearest-min)
# + R=1 circular-correlation (free-field) & structural-zero helpers
# ================================================================
# - Free-field 2πk resonance + structural zeros (k in 3Z: theoretical)
# - e-Operator with Δlog p target pairing + period-aware coarse scan
#   + "nearest-local-min" bracket scan + Brent (no boundary lock)
# - Optional Λ-direct check (prime-only baseline)
# - Zero-mean kernel & weighting mode switches
# - Pure matplotlib + print; no file writes
# ================================================================

# 📦 Setup
!pip -q install sympy numpy scipy tqdm

import numpy as np
import math
from sympy import primerange
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from tqdm import tqdm
from bisect import bisect_left

# =========================
# Global Parameters (easy)
# =========================
M = 100_000                  # 100k 起步；200k 更重
alpha = 1.0                  # 自由场 J_free 系数
differences = [1, 2, 6, 30]  # 自由场小差集 H
seed = 123
np.random.seed(seed)

# 自由场局部细扫
free_local_steps = 4_001

# e-Operator（Δlog p 目标配对）
DELTA_MIN, DELTA_MAX = 0.2, 3.0   # Δlog p 覆盖（到 3.0 更宽）
N_DELTAS = 60                     # 目标采样数（40~60）
I_STRIDE = 3                      # i 步幅（1=全量；2/3 降耗）

# e-Operator：τ 粗扫范围（周期感知会自动设点数）
tau_scan_min, tau_scan_max = 0.0, 60.0
tau_scan_points_min = 8_000       # 最低点数；周期感知会增大
period_pts = 80                   # 每周期采样点数（60~100）

# e-Operator 核与权重选项
USE_DEMEAN_KERNEL = True          # True 让极小更锐
WEIGHTING_MODE = "none"           # "delta_p" 或 "none" （推荐 "none" 更均衡）

# Λ-direct（True=跑；False=跳过）
RUN_LAMBDA_DIRECT = True

# Brent 精细化（初窗≈半周期，适合 period ~ 3-6）
BRENT_W0 = 1.8
BRENT_EXPAND = 0.6
BRENT_MAXEXP = 8
BRENT_HMIN = 1e-3

# ζ 零点（前 10 个）
zeta_zeros_im = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876125,
    32.935061588, 37.586178159, 40.918719012, 43.327073281,
    48.005150881, 49.773832478
])

print(f"[SETUP] M={M}, H={differences}, alpha={alpha}, "
      f"DELTAs=[{DELTA_MIN},{DELTA_MAX}], N_DELTAS={N_DELTAS}, I_STRIDE={I_STRIDE}, "
      f"demean={USE_DEMEAN_KERNEL}, weight={WEIGHTING_MODE}")

# ==========================
# Generate first M primes
# ==========================
print("[PRIMES] Generating primes...")
primes = []
upper_bound = 30_000_000  # 对 100k 足够；更大 M 可放宽
for p in primerange(2, upper_bound):
    primes.append(int(p))
    if len(primes) >= M:
        break
primes = np.array(primes, dtype=np.int64)
logp = np.log(primes).astype(np.float64)

# ==========================
# FREE-FIELD graph (H = {1,2,6,30})
# ==========================
H = set(differences)
H_pos = {6, 30}
H_neg = {1, 2}
index_by_value = {v: i for i, v in enumerate(primes)}
rows_D, cols_D, data_D = [], [], []
w_e, s_e = [], []

edge_count = 0
for i, pi in enumerate(primes):
    for h in H:
        pj = pi + h
        j = index_by_value.get(pj, None)
        if j is None:
            continue
        rows_D += [edge_count, edge_count]
        cols_D += [i, j]
        data_D += [-1.0, +1.0]
        wij = 1.0 / (logp[i] * logp[j])
        w_e.append(wij)
        s_e.append(+1.0 if h in H_pos else -1.0)
        edge_count += 1

E = edge_count
w_e = np.array(w_e, dtype=np.float64)
s_e = np.array(s_e, dtype=np.float64)

print(f"[GRAPH/FREE] |V|={M}, |E|={E}")
if E == 0:
    raise RuntimeError("No FREE-FIELD edges; increase M or adjust H.")

D = coo_matrix((data_D, (rows_D, cols_D)), shape=(E, M)).tocsr()
W = diags(w_e)
L = D.T @ W @ D
print(f"[LAPLACIAN] shape={L.shape}, nnz={L.nnz}")

# 预条件探针 v（更稳），也可换随机 v
r = np.random.normal(size=M)
u = lsqr(L, r, atol=1e-10, btol=1e-10, iter_lim=200)[0]
v = u / (np.linalg.norm(u) + 1e-15)

def Sop_times_vec(v):
    x = D @ v
    x = s_e * w_e * x
    return D.T @ x

Sop_v = Sop_times_vec(v)
norm_Sop_v = np.linalg.norm(Sop_v)
print(f"[NORM] ||S_op v||_2 = {norm_Sop_v:.6e}")

# -----------------------------
# FREE objective + utilities
# -----------------------------

def J_free(tau):
    return alpha * abs(1.0 - math.cos(tau)) * norm_Sop_v


def local_scan_free(k, span=0.02, steps=4_001):
    tau0 = 2 * math.pi * k
    tt = np.linspace(tau0 - span, tau0 + span, steps)
    jj = np.array([J_free(t) for t in tt])
    i = np.argmin(jj)
    return tau0, tt[i], jj[i], tt, jj

# -----------------------------
# NEW: R=1 computation (free)
# -----------------------------

def compute_R_and_deltas_free(K_max=6, span=0.02, steps=4001):
    deltas = []
    records = []
    for k in range(1, K_max + 1):
        tau0, tmin, jmin, _, _ = local_scan_free(k, span=span, steps=steps)
        delta = tmin - tau0
        deltas.append(delta)
        records.append((k, tmin, delta, jmin))
    deltas = np.array(deltas)
    R = np.abs(np.mean(np.exp(1j * deltas)))
    return R, records

# -----------------------------
# NEW: structural zeros helper
# -----------------------------
# 说明：在 FREE 目标下，所有 k 都在 τ=2πk 取得 J=0（平凡零）。
# 若只想报告“理论结构零点”为 k∈3Z：使用 rule-based 判定。
# 若想做数值判定（排除中心点），用“穿孔邻域”的最小值作为对比；
# 但对 FREE 目标它不会区分 k（因为是二次近似且与 k 无关）。


def structural_zero_ks_theoretical(K_max=12):
    return [k for k in range(1, K_max + 1) if k % 3 == 0]


# 可选：穿孔邻域数值测试（对 e-operator 更有区分力）

def punctured_min_free(k, eps=1e-6, span=2e-2, steps=4001):
    tau0 = 2 * math.pi * k
    tt = np.linspace(tau0 - span, tau0 + span, steps)
    mask = np.abs(tt - tau0) >= eps
    tt = tt[mask]
    jj = np.array([J_free(t) for t in tt])
    i = int(np.argmin(jj))
    return float(tt[i]), float(jj[i])

# =============================
# Free-field: plots + printing
# =============================

# 全局粗扫（展示 2πk）
tau_max = 12 * math.pi
num_pts = 3000
taus_free = np.linspace(0.0, tau_max, num_pts)
Jvals_free = np.array([J_free(t) for t in taus_free])

plt.figure()
plt.plot(taus_free, Jvals_free)
plt.xlabel("τ")
plt.ylabel("J(τ) (free-field)")
plt.title("Free-field J(τ): minima at τ = 2πk")
plt.grid(True)
plt.show()

print("[FREE] Nearest-sample minima around 2πk:")
for k in range(1, int(tau_max / (2 * math.pi)) + 1):
    tau0 = 2 * math.pi * k
    idx = np.argmin(np.abs(taus_free - tau0))
    print(f"  k={k:2d}, τ≈{taus_free[idx]:.6f}, 2πk={tau0:.6f}, Δ={(taus_free[idx]-tau0):+.3e}, J≈{Jvals_free[idx]:.3e}")

print("\n[FREE] Local refinement around 2πk (k=3 and k=6):")
for kk in [3, 6]:
    tau0, tmin, jmin, _, _ = local_scan_free(kk, span=0.02, steps=free_local_steps)
    print(f"  k={kk}, 2πk={tau0:.6f}, τ*={tmin:.9f}, Δ={tmin - tau0:+.3e}, J*={jmin:.3e}")

# === NEW: compute R and structural zeros (free) ===
R, recs = compute_R_and_deltas_free(K_max=6, span=0.02, steps=free_local_steps)
print("\n[FREE] Circular correlation (R) & refined minima:")
for (k, tmin, delta, jmin) in recs:
    print(f"k={k:2d}  tau*={tmin:.9f}  Δ={delta:+.3e}  J*={jmin:.3e}")
print("R=", R)

print("structural-zero ks (theoretical k∈3Z):", structural_zero_ks_theoretical(12))

# =========================================================
# e-Operator: Δlog p 目标配对（覆盖远近尺度）
# =========================================================

delta_targets = np.linspace(DELTA_MIN, DELTA_MAX, N_DELTAS)


def find_j_for_delta(i, d):
    # 目标：p_j ≈ p_i * e^d
    target = primes[i] * math.exp(d)
    j = bisect_left(primes, target, lo=i + 1, hi=len(primes))
    if j >= len(primes):
        return None
    if j == i + 1:
        return j
    prev = j - 1
    if prev <= i:
        return j
    return j if abs(primes[j] - target) < abs(primes[prev] - target) else prev


ij_i, ij_j, w0 = [], [], []
for i in range(0, len(primes), I_STRIDE):
    for d in delta_targets:
        j = find_j_for_delta(i, float(d))
        if (j is None) or (j <= i):
            continue
        wij = 1.0 / (logp[i] * logp[j])  # 对称权
        ij_i.append(i)
        ij_j.append(j)
        w0.append(wij)

ij_i = np.array(ij_i, dtype=np.int32)
ij_j = np.array(ij_j, dtype=np.int32)
w0 = np.array(w0, dtype=np.float64)

logdiff = (logp[ij_j] - logp[ij_i]).astype(np.float64)
deltap = (primes[ij_j] - primes[ij_i]).astype(np.float64)
if WEIGHTING_MODE == "delta_p":
    baseline = np.linalg.norm(w0 * np.abs(deltap)) + 1e-15
else:
    baseline = np.linalg.norm(w0) + 1e-15

print(
    f"\n[E-OP/EDGES: Δlog p targets] pairs={len(w0)}, "
    f"Δlog p in [{logdiff.min():.3f}, {logdiff.max():.3f}] "
    f"(median {np.median(logdiff):.3f})"
)

# e-operator 目标

def J_eop(tau):
    s2 = np.sin(tau * logdiff)
    s2 *= s2
    if USE_DEMEAN_KERNEL:
        s2 -= np.mean(s2)
    if WEIGHTING_MODE == "delta_p":
        diff = (w0 * s2) * np.abs(deltap)
    else:
        diff = w0 * s2
    return np.linalg.norm(diff) / baseline

# ========= 周期感知的粗扫（以 Δlog p 的中位数估计） =========
mu = np.median(np.abs(logdiff)) + 1e-12
suggested_period = math.pi / mu

tau_scan_points = max(
    tau_scan_points_min,
    int((tau_scan_max - tau_scan_min) / (suggested_period / period_pts))
)

taus = np.linspace(tau_scan_min, tau_scan_max, tau_scan_points)
print(f"[E-OP/SCAN] period≈{suggested_period:.3f}, scan_points={tau_scan_points}")

Jz = np.array([J_eop(t) for t in tqdm(taus, desc="e-operator coarse scan (Δlog p targets)")])

plt.figure()
plt.plot(taus, Jz)
plt.xlabel("τ")
plt.ylabel("J(τ) with pure e-operator (normalized)")
plt.title("e-Operator (pure) scan — Δlog p target pairing")
plt.grid(True)
plt.show()

# ========= 就近局部极小：先“扫描找括号（最近 b）”再 Brent =========


def _find_bracket_by_scan_near(f, z, w, h_min=1e-3, max_points=2001):
    """
    在 [z-w, z+w] 扫描，收集所有三点凹口 (a<b<c 且 f(b)<=f(a), f(b)<=f(c))，
    返回“离 z 最近”的那个括号。
    """
    a = max(0.0, z - w)
    c = z + w
    if c - a < 3 * h_min:
        c = a + 3 * h_min
    n = max(5, min(max_points, int((c - a) / h_min)))
    xs = np.linspace(a, c, n)
    ys = np.array([f(x) for x in xs])

    candidates = []
    for i in range(1, n - 1):
        if ys[i] <= ys[i - 1] and ys[i] <= ys[i + 1]:
            candidates.append((xs[i - 1], xs[i], xs[i + 1]))

    if not candidates:
        return None

    b_list = np.array([b for (_, b, _) in candidates])
    k = int(np.argmin(np.abs(b_list - z)))
    return candidates[k]


def refine_min_brent_robust(
    f, z, taus, Jz, w0=BRENT_W0, expand_step=BRENT_EXPAND, max_expand=BRENT_MAXEXP, h_min=BRENT_HMIN
):
    # 先“就近”找括号，再 Brent；找不到则 bounded 兜底 + 自适应扩窗
    for k in range(max_expand + 1):
        w = w0 + k * expand_step
        br = _find_bracket_by_scan_near(f, z, w, h_min=h_min, max_points=2001)
        if br is not None:
            a, b, c = br
            res = minimize_scalar(
                f, method='brent', bracket=(a, b, c), options={'xtol': 1e-9, 'maxiter': 500}
            )
            if res.success:
                t, j = float(res.x), float(res.fun)
                if abs(t - a) < 1e-3 or abs(t - c) < 1e-3:  # 贴边再扩窗
                    continue
                return t, j, (a, b, c)
    # 兜底：bounded + 扩窗避免“卡边界”
    for k in range(max_expand + 1):
        w = w0 + k * expand_step
        a = max(0.0, z - w)
        c = z + w
        res = minimize_scalar(
            f, bounds=(a, c), method='bounded', options={'xatol': 1e-9, 'maxiter': 500}
        )
        if res.success:
            t, j = float(res.x), float(res.fun)
            if abs(t - a) < 1e-3 or abs(t - c) < 1e-3:
                continue
            return t, j, (a, (a + c) / 2.0, c)
    # 最后用粗扫最小
    a = max(0.0, z - (w0 + max_expand * expand_step))
    c = z + (w0 + max_expand * expand_step)
    m = (taus >= a) & (taus <= c)
    if np.any(m):
        i = np.argmin(Jz[m])
        return float(taus[m][i]), float(Jz[m][i]), (a, (a + c) / 2.0, c)
    return None, None, (a, (a + c) / 2.0, c)

print("\n[E-OP/PURE] Robust Brent refinement (nearest-local-min):")
print("   target_tau     tau*        J(tau*)        Δ=tau*-target   bracket")
refined = []
for z in zeta_zeros_im:
    t_ref, j_ref, br = refine_min_brent_robust(J_eop, z, taus, Jz)
    refined.append((z, t_ref, j_ref))
    if t_ref is None:
        print(
            f" {z:11.6f}   (none)      (none)         (none)           {tuple(round(x, 3) for x in br)}"
        )
    else:
        a, b, c = br
        print(
            f" {z:11.6f}  {t_ref:9.6f}   {j_ref:11.3e}   {t_ref - z:+.6f}   {(round(a, 3), round(b, 3), round(c, 3))}"
        )

# 叠加可视化：粗扫 + ζ 零点 + Brent
plt.figure()
plt.plot(taus, Jz, label="J(τ) coarse")
for z in zeta_zeros_im:
    plt.axvline(z, linestyle="--", linewidth=1, label=None)
xs = [t for (_, t, _) in refined if t is not None]
ys = [j for (_, _, j) in refined if j is not None]
plt.scatter(xs, ys, marker="o", s=30, zorder=3, label="Brent minima")
plt.xlabel("τ")
plt.ylabel("J(τ) with pure e-operator (normalized)")
plt.title("e-Operator J(τ): ζ-zeros (dashed) vs refined minima (dots)")
plt.grid(True)
plt.legend()
plt.show()

# 统计误差
rows = []
print("\n[E-OP/PURE] Nearest-to-z minima (stats):")
print("   target_tau     tau*        Δ=tau*-target")
for (z, t_ref, j_ref) in refined:
    if t_ref is None:
        print(f" {z:11.6f}   (none)       (none)")
    else:
        d = t_ref - z
        rows.append(abs(d))
        print(f" {z:11.6f}  {t_ref:9.6f}   {d:+.6f}")
if rows:
    arr = np.array(rows)
    print("\n[STATS] mean|Δ| = %.6f, median|Δ| = %.6f, max|Δ| = %.6f" % (arr.mean(), np.median(arr), arr.max()))

# ==========================
# Optional: Λ-direct (prime-only) 对照（零均值核更锐）
# ==========================
if RUN_LAMBDA_DIRECT:
    normL = np.linalg.norm(logp) + 1e-15

    def J_vm(tau):
        s2 = np.sin(tau * logp)
        s2 *= s2
        if USE_DEMEAN_KERNEL:
            s2 -= np.mean(s2)  # 去 DC
        return np.sqrt(np.sum((logp * s2) ** 2)) / normL

    Jz_vm = np.array([J_vm(t) for t in taus])  # 仅作兜底
    print("\n[Λ-DIRECT] Robust Brent refinement (nearest-local-min):")
    print("   target_tau     tau*        J(tau*)        Δ=tau*-target   bracket")
    refined_vm = []
    for z in zeta_zeros_im:
        t_ref, j_ref, br = refine_min_brent_robust(J_vm, z, taus, Jz_vm)
        refined_vm.append((z, t_ref, j_ref))
        if t_ref is None:
            print(
                f" {z:11.6f}   (none)      (none)         (none)           {tuple(round(x, 3) for x in br)}"
            )
        else:
            a, b, c = br
            print(
                f" {z:11.6f}  {t_ref:9.6f}   {j_ref:11.3e}   {t_ref - z:+.6f}   {(round(a, 3), round(b, 3), round(c, 3))}"
            )

# ===================
# Spectrum (optional) — skipped to avoid long runs
# ===================
print("\n[SPECTRUM] skipped for speed at M=100k.")
print("\n[DONE] Free-field 2πk + e-operator (Δlog p pairing) completed.")
print("Tips: if offsets remain, widen DELTA_MAX→3.0, set N_DELTAS→60,")
print("      USE_DEMEAN_KERNEL=True, WEIGHTING_MODE='none', lower I_STRIDE,")
print("      and keep period_pts≈80–100 for dense coarse scan.")
# ----------------------------------------------------------
# STRUCTURAL ZEROS via e-operator antisymmetric probe  A(τ)
# ----------------------------------------------------------
def antisym_probe(tau):
    # 反对称（正弦）探针，不做去均值；只衡量抵消程度
    return abs(np.sum(w0 * np.sin(2.0 * tau * logdiff)))

K_max = 12  # 看前 12 个 k
A_vals = []
for k in range(1, K_max + 1):
    tau_k = 2.0 * math.pi * k
    A_k = antisym_probe(tau_k)
    A_vals.append((k, tau_k, A_k))

# 归一化后判定：小于某个分位数/阈值视为结构零点
vals = np.array([x[2] for x in A_vals])
# 用稳健阈值（例如 25% 分位数的 0.75 倍）；你也可以固定阈值，如 1e-3
thr = 0.75 * np.quantile(vals, 0.25)
structural_ks = [k for (k, _, A_k) in A_vals if A_k <= thr]

print("\n[STRUCT-ZEROS / e-operator antisym probe]")
for (k, tau_k, A_k) in A_vals:
    tag = " **STRUCT**" if k in structural_ks else ""
    print(f"k={k:2d}  τ=2πk={tau_k:9.6f}   A(τ)={A_k:.3e}{tag}")
print("Detected structural-zero ks:", structural_ks)
