# ===================== Core-Verification Colab (ANP+Prime Alignment & Asymptotic Scaling) =====================
!pip -q install sympy scipy

import numpy as np, math, matplotlib.pyplot as plt, pandas as pd
from sympy import primerange
from scipy.optimize import minimize_scalar

np.set_printoptions(suppress=True, precision=6)

# --------------------------- Toggles ---------------------------
RUN_PER_ZERO      = True     # 核心1：逐零点对齐（ANP+Prime 轴）
RUN_SCALING       = True     # 核心2：误差随 M 的幂律缩放（含 bootstrap CI）
RUN_CONTROLS      = True     # 对照：随机 log / 奇数
SHOW_PLOTS        = True     # 画图（matplotlib）

# 局部极小搜索参数
PROFILE     = "DEEP"         # "FAST" / "DEEP"（影响局部粗扫点数）
WINDOW_TAU  = 1.2            # 零点邻域窗口（τ 轴）
GRID_LOCAL  = 401 if PROFILE=="DEEP" else 201
TRIM_FRAC   = 0.20           # 20% 截尾

# ζ 零点（前 10 个 imag 部分）
Z_IM = np.array([
    14.134725141, 21.022039639, 25.010857580, 30.424876125,
    32.935061588, 37.586178159, 40.918719012, 43.327073281,
    48.005150881, 49.773832478
], dtype=float)

# 缩放实验的 M 列表（可按需增大到 6400/10000 等）
MS = [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]

# --------------------------- Primes & helpers ---------------------------
def build_primes(M):
    """返回前 M 个素数及其对数。"""
    ps = []
    for p in primerange(2, 10_000_000):
        ps.append(int(p))
        if len(ps) >= M: break
    P = np.array(ps, dtype=np.int64); LP = np.log(P).astype(float)
    return P, LP

def J_Lambda0_factory(logp):
    """Prime 轴 von Mangoldt 零条件：J_{Λ0}(τ) = |∑ log p * sin(2 τ log p)| / ∑ log p"""
    S = float(np.sum(logp) + 1e-15)
    def J_Lambda0(tau: float) -> float:
        s = np.sin(2.0 * tau * logp)
        return abs(float(np.sum(logp * s))) / S
    return J_Lambda0

def find_local_minima(xs, ys):
    idxs = []
    for i in range(1, len(xs)-1):
        if ys[i] <= ys[i-1] and ys[i] <= ys[i+1]:
            idxs.append(i)
    return np.array(idxs, dtype=int)

def refine_minimum(func, center, window=1.2, grid_points=201, xtol=1e-9):
    """在 [center±window] 内粗扫找极小，再用 Brent 精化。"""
    a = max(0.0, center - window); c = center + window
    xs = np.linspace(a, c, grid_points)
    ys = np.array([func(x) for x in xs])
    mins = find_local_minima(xs, ys)
    if len(mins) == 0:
        res = minimize_scalar(func, bounds=(a, c), method="bounded",
                              options={"xatol": xtol, "maxiter": 500})
        return float(res.x), float(res.fun)
    midx = mins[np.argmin(np.abs(xs[mins] - center))]
    L = max(midx-1, 0); R = min(midx+1, len(xs)-1)
    a2, b2, c2 = float(xs[L]), float(xs[midx]), float(xs[R])
    res = minimize_scalar(func, method="brent", bracket=(a2, b2, c2),
                          options={"xtol": xtol, "maxiter": 500})
    return float(res.x), float(res.fun)

def trimmed_mean(errs, trim_frac=0.2):
    n = len(errs)
    if n == 0: return float('nan')
    k = int(trim_frac * n)
    if 2*k >= n: return float(np.mean(errs))
    trimmed = np.sort(errs)[k:n-k]
    return float(np.mean(trimmed))

def fit_powerlaw(x, y):
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    return float(slope), float(math.exp(intercept))  # y ≈ C * x^slope

def bootstrap_slope_CI(Ms, vals, B=1000, seed=2025):
    """对 (M, y) 点做行引导（resample points）估计斜率 CI。"""
    rng = np.random.default_rng(seed)
    Ms, vals = np.array(Ms, float), np.array(vals, float)
    mask = vals > 0
    Ms, vals = Ms[mask], vals[mask]
    if len(Ms) < 3:
        return (float('nan'), float('nan')), (float('nan'), float('nan'))
    slopes = []
    for _ in range(B):
        idx = rng.integers(0, len(Ms), size=len(Ms))
        s, _ = fit_powerlaw(Ms[idx], vals[idx])
        slopes.append(s)
    slopes = np.sort(np.array(slopes))
    lo, hi = slopes[int(0.025*B)], slopes[int(0.975*B)]
    s_hat, _ = fit_powerlaw(Ms, vals)
    return s_hat, (lo, hi)

# ===================== Part A: 核心1 — 编码/对齐（逐零点） =====================
if RUN_PER_ZERO:
    M0 = 1200  # 可调：用于展示逐零点对齐的 M
    print(f"=== Part A: ANP + Prime Axis Alignment (per-zero at M={M0}) ===")
    _, logp = build_primes(M0)
    JLam = J_Lambda0_factory(logp)

    rows = []
    for t in Z_IM:
        tau_target = t/2.0
        tau_star, j_star = refine_minimum(JLam, tau_target, window=WINDOW_TAU, grid_points=GRID_LOCAL)
        err = abs(t - 2.0*tau_star)
        rows.append((t, tau_target, tau_star, err, j_star))
        print(f"t={t:10.6f}  tau≈t/2={tau_target:9.6f}  tau*={tau_star:9.6f}  |t-2τ*|={err:7.4e}  J*={j_star:6.2e}")

    errs = np.array([r[3] for r in rows], float)
    print(f"\n[Prime axis J_{{Λ0}} :: |t-2τ*| (M={M0})]")
    print(f"  median          = {np.median(errs):.6f}")
    print(f"  trimmed mean20% = {trimmed_mean(errs, TRIM_FRAC):.6f}")
    print(f"  max             = {np.max(errs):.6f}")

    if SHOW_PLOTS:
        plt.figure(figsize=(8,4))
        plt.bar(np.arange(len(errs)), errs)
        plt.xticks(np.arange(len(errs)), [f"{i+1}" for i in range(len(errs))])
        plt.ylabel("|t - 2 τ*|")
        plt.xlabel("Zero index (first 10)")
        plt.title(f"Per-zero alignment error at M={M0} (Prime axis)")
        plt.show()

# ===================== Part B: 核心2 — 渐近精确（缩放） =====================
if RUN_SCALING:
    print("\n=== Part B: Asymptotic Exactness (scaling of error vs M) ===")
    results = []
    for M in MS:
        _, logp = build_primes(M)
        JL = J_Lambda0_factory(logp)
        errs = []
        for t in Z_IM:
            tau_target = t/2.0
            tau_star, j_star = refine_minimum(JL, tau_target, window=WINDOW_TAU, grid_points=GRID_LOCAL)
            errs.append(abs(t - 2.0*tau_star))
        errs = np.array(errs, float)
        results.append((M, float(np.median(errs)), trimmed_mean(errs, TRIM_FRAC), float(np.max(errs))))

    df = pd.DataFrame(results, columns=["M","median","tmean20","max"])
    print(df.to_string(index=False))

    # 幂律斜率 + bootstrap CI
    s_med, ci_med = bootstrap_slope_CI(df["M"].values, df["median"].values, B=1000, seed=123)
    s_tmn, ci_tmn = bootstrap_slope_CI(df["M"].values, df["tmean20"].values, B=1000, seed=456)
    print("\n=== Log–log slope (with 95% bootstrap CI) ===")
    print(f"median  slope ≈ {s_med:.3f}  CI[{ci_med[0]:.3f}, {ci_med[1]:.3f}]")
    print(f"tmean20 slope ≈ {s_tmn:.3f}  CI[{ci_tmn[0]:.3f}, {ci_tmn[1]:.3f}]")

    verdict = "SUPPORTED" if ((not np.isnan(s_med) and ci_med[1] < 0.0) or (not np.isnan(s_tmn) and ci_tmn[1] < 0.0)) else "WEAK / INCONCLUSIVE"
    print(f"\n[Verdict] Asymptotic Exactness is {verdict} under current settings.")

    if SHOW_PLOTS:
        # median 曲线 + 拟合线
        mask = df["median"].values > 0
        Ms_fit = df["M"].values[mask].astype(float)
        y_med  = df["median"].values[mask].astype(float)
        s_med_fit, C_med_fit = fit_powerlaw(Ms_fit, y_med)
        plt.figure(figsize=(6,4))
        plt.loglog(df["M"].values, df["median"].values, marker='o', label='median')
        Ms_line = np.linspace(min(Ms_fit), max(Ms_fit), 200)
        plt.loglog(Ms_line, C_med_fit * Ms_line**s_med_fit, linestyle='--', label=f'fit slope≈{s_med_fit:.3f}')
        plt.xlabel("M"); plt.ylabel("median |t - 2 τ*|")
        plt.title("Scaling of median error vs M (Prime axis)")
        plt.legend(); plt.show()

        # trimmed mean 曲线 + 拟合线
        mask = df["tmean20"].values > 0
        Ms_fit = df["M"].values[mask].astype(float)
        y_tmn  = df["tmean20"].values[mask].astype(float)
        s_tmn_fit, C_tmn_fit = fit_powerlaw(Ms_fit, y_tmn)
        plt.figure(figsize=(6,4))
        plt.loglog(df["M"].values, df["tmean20"].values, marker='o', label='tmean20')
        Ms_line = np.linspace(min(Ms_fit), max(Ms_fit), 200)
        plt.loglog(Ms_line, C_tmn_fit * Ms_line**s_tmn_fit, linestyle='--', label=f'fit slope≈{s_tmn_fit:.3f}')
        plt.xlabel("M"); plt.ylabel("trimmed-mean (20%) |t - 2 τ*|")
        plt.title("Scaling of trimmed-mean error vs M (Prime axis)")
        plt.legend(); plt.show()

# ===================== Part C: 对照试验（可选） =====================
if RUN_CONTROLS:
    print("\n=== Part C: Controls (random logs / odd integers) at M=1200 ===")
    Mctrl = 1200
    _, logp = build_primes(Mctrl)
    JL = J_Lambda0_factory(logp)

    # 控制1：随机 log（与素数无关的独立同分布）
    rng = np.random.default_rng(7)
    log_rand = rng.normal(size=Mctrl); log_rand = np.abs(log_rand) + 1.0  # positive
    JL_rand = J_Lambda0_factory(log_rand)

    # 控制2：奇数序列（3,5,7,...）
    odds = np.arange(3, 3+2*Mctrl, 2, dtype=int)
    log_odds = np.log(odds).astype(float)
    JL_odds = J_Lambda0_factory(log_odds)

    def run_control(Jfunc, label):
        errs = []
        for t in Z_IM:
            tau_target = t/2.0
            tau_star, _ = refine_minimum(Jfunc, tau_target, window=WINDOW_TAU, grid_points=GRID_LOCAL)
            errs.append(abs(t - 2.0*tau_star))
        errs = np.array(errs, float)
        print(f"{label:>12s}  median={np.median(errs):.6f}  tmean20={trimmed_mean(errs, TRIM_FRAC):.6f}  max={np.max(errs):.6f}")
        return errs

    errs_primes = run_control(JL,      "primes(M=1200)")
    errs_rand   = run_control(JL_rand, "random-log")
    errs_odds   = run_control(JL_odds, "odd-integers")

    if SHOW_PLOTS:
        plt.figure(figsize=(8,4))
        x = np.arange(3)
        y = [np.median(errs_primes), np.median(errs_rand), np.median(errs_odds)]
        plt.bar(x, y)
        plt.xticks(x, ["primes", "rand-log", "odds"])
        plt.ylabel("median |t - 2 τ*|")
        plt.title("Controls (M=1200): lower is better")
        plt.show()

# ===================== Summary =====================
print("\n=== Summary ===")
print("* 核心1（编码/对齐）：Part A 打印逐零点 (tau*, |t-2τ*|) 及统计；")
print("* 核心2（渐近精确）：Part B 给出误差~M 的幂律斜率及 95% bootstrap CI，并画 log–log 图；")
print("* 对照：Part C 比较 primes vs. random-log / odd-integers（应当 primes 更优或同级，显示结构差异）；")
print("把 Part A 的逐零点表 + Part B 的缩放表/图 直接放入论文验证章节即可。")
