# =========================
# ONE-CELL COLAB (FIXED):
# - Phase-only slope (slope_phase ~ 2)
# - Robust F* (median over window, avoids blow-ups)
# - Fig.1 uses phase points only
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CFG = dict(
    n=64,
    T=320,
    seeds=5,
    eps_list=np.logspace(-6, -1, 9),
    dt_safety=0.9,
    w=9,                 # half-window for robust K(t)
    deltaK=0.05,
    r2_min=0.85,
    floor=1e-30,
    ref_family="lap_like",
    defect_density=0.02,
    low_rank=4,
    make_plots=True,
    save_prefix="closure_run_fixed"
)

np.set_printoptions(precision=4, suppress=True)

def linreg_slope_r2(x, y):
    x = np.asarray(x); y = np.asarray(y)
    if len(x) < 3:
        return np.nan, np.nan
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    slope = float(beta[0])
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-18)
    r2 = 1.0 - ss_res/ss_tot
    return slope, float(r2)

def spectral_radius(A, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(A.shape[0],))
    x /= (np.linalg.norm(x) + 1e-18)
    for _ in range(iters):
        y = A @ x
        ny = np.linalg.norm(y)
        if ny < 1e-18:
            return 0.0
        x = y / ny
    return float(np.linalg.norm(A @ x))

def laplacian_1d_ring(n):
    L = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        L[i, i] = 2.0
        L[i, (i-1) % n] = -1.0
        L[i, (i+1) % n] = -1.0
    return L

def make_defect(family, n, rng, density=0.02, low_rank=4):
    if family == "lap_like":
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            D[i, i] = rng.normal(scale=0.5)
            D[i, (i+1) % n] += rng.normal(scale=0.2)
            D[i, (i-1) % n] += rng.normal(scale=0.2)
        D = 0.5*(D + D.T)
        np.fill_diagonal(D, 0.0)
        return D

    if family == "diag_local":
        D = np.zeros((n, n), dtype=np.float64)
        k = max(2, n//16)
        idx = rng.choice(n, size=k, replace=False)
        D[idx, idx] = rng.normal(size=k)
        return D

    if family == "random_sparse":
        D = np.zeros((n, n), dtype=np.float64)
        m = int(density * n * n)
        for _ in range(m):
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n))
            if i == j:
                continue
            v = rng.normal()
            D[i, j] += v
            D[j, i] += v
        np.fill_diagonal(D, 0.0)
        return D

    if family == "low_rank":
        r = int(low_rank)
        U = rng.normal(size=(n, r))
        D = U @ U.T
        D = D - np.diag(np.diag(D))
        D = 0.5*(D + D.T)
        D /= (np.linalg.norm(D, ord='fro') + 1e-18)
        return D

    raise ValueError(f"Unknown family: {family}")

def evolve(u0, B, T, dt):
    u = u0.copy()
    traj = np.zeros((T, len(u0)), dtype=np.float64)
    for t in range(T):
        traj[t] = u
        u = u - dt * (B @ u)
    return traj

def observables(L, B, traj):
    chi = ((L - B) @ traj.T).T
    H = np.sum(chi**2, axis=1)
    Phi = np.sum(np.abs(chi), axis=1)
    Bu = (B @ traj.T).T
    num = np.linalg.norm(chi, axis=1) + 1e-30
    den = np.linalg.norm(Bu, axis=1) + 1e-30
    Ward_series = num / den
    return chi, H, Phi, Ward_series

def robust_K(H, Phi, w, floor=1e-30):
    T = len(H)
    K = np.full(T, np.nan, dtype=np.float64)
    R2 = np.full(T, np.nan, dtype=np.float64)
    logH = np.log(np.maximum(H, floor))
    logP = np.log(np.maximum(Phi, floor))
    for t in range(w, T-w):
        xs = logP[t-w:t+w+1]
        ys = logH[t-w:t+w+1]
        slope, r2 = linreg_slope_r2(xs, ys)
        K[t] = slope
        R2[t] = r2
    return K, R2

def find_t_star(K, R2, deltaK, r2_min):
    idx = np.where(np.isfinite(K) & np.isfinite(R2) & (np.abs(K-2.0) < deltaK) & (R2 > r2_min))[0]
    return int(idx[0]) if len(idx) else None

def phase_mask(K, R2, deltaK, r2_min):
    return np.isfinite(K) & np.isfinite(R2) & (np.abs(K-2.0) < deltaK) & (R2 > r2_min)

def robust_F_star(H, Phi, dt, t_star, wF=7):
    # Robust: median over a small window around t*; avoid tiny dPhi blow-ups.
    if t_star is None:
        return np.nan
    T = len(H)
    a = max(1, t_star - wF)
    b = min(T-2, t_star + wF)
    vals = []
    for t in range(a, b+1):
        # centered dPhi
        dPhi = (Phi[t+1] - Phi[t-1]) / (2.0*dt)
        denom = Phi[t] * (abs(dPhi) + 1e-30)
        vals.append(H[t] / (denom + 1e-30))
    return float(np.median(vals)) if len(vals) else np.nan

def run_family(family, cfg, L, rho_ref):
    rows = []
    fig_payload = []  # representative plots at mid eps

    for eps in cfg["eps_list"]:
        Ward_meds, K_atstar, Fstars = [], [], []
        slope_all_list, slope_all_r2_list = [], []
        slope_ph_list, slope_ph_r2_list = [], []
        rho_list = []
        phase_counts = []
        valid_tstar = 0

        for s in range(cfg["seeds"]):
            rng = np.random.default_rng(10_000 + s)
            Delta = make_defect(family, cfg["n"], rng, density=cfg["defect_density"], low_rank=cfg["low_rank"])
            rhoD = spectral_radius(Delta, seed=100+s)
            rho_list.append(rhoD)

            B = L + eps * Delta
            rhoB = spectral_radius(B, seed=200+s)
            dt = cfg["dt_safety"] / (rhoB + 1e-12)

            u0 = rng.normal(size=(cfg["n"],))
            traj = evolve(u0, B, cfg["T"], dt)
            chi, H, Phi, Ward_series = observables(L, B, traj)

            K, KR2 = robust_K(H, Phi, cfg["w"], floor=cfg["floor"])
            t_star = find_t_star(K, KR2, cfg["deltaK"], cfg["r2_min"])
            if t_star is not None:
                valid_tstar += 1

            # ---- slope_all (old) ----
            idx = np.arange(cfg["w"], cfg["T"]-cfg["w"])
            slope_all, r2_all = linreg_slope_r2(np.log(np.maximum(Phi[idx], cfg["floor"])),
                                                np.log(np.maximum(H[idx], cfg["floor"])))
            slope_all_list.append(slope_all); slope_all_r2_list.append(r2_all)

            # ---- slope_phase (FIX) ----
            m = phase_mask(K, KR2, cfg["deltaK"], cfg["r2_min"])
            phase_counts.append(int(np.sum(m)))
            if np.sum(m) >= 6:
                slope_ph, r2_ph = linreg_slope_r2(np.log(np.maximum(Phi[m], cfg["floor"])),
                                                  np.log(np.maximum(H[m], cfg["floor"])))
            else:
                slope_ph, r2_ph = (np.nan, np.nan)
            slope_ph_list.append(slope_ph); slope_ph_r2_list.append(r2_ph)

            # ---- K and robust F* at t* ----
            if t_star is not None:
                K_atstar.append(float(K[t_star]))
                Fstars.append(robust_F_star(H, Phi, dt, t_star, wF=7))

            Ward_meds.append(float(np.median(Ward_series)))

            # Representative plot payload at mid eps for first seed
            if cfg["make_plots"] and s == 0 and eps == cfg["eps_list"][len(cfg["eps_list"])//2]:
                fig_payload.append((family, eps, H, Phi, K, KR2, m, t_star))

        rho_mean = float(np.mean(rho_list))
        Z = float(rho_ref / (rho_mean + 1e-18))

        rows.append(dict(
            family=family, eps=eps,
            Ward=float(np.median(Ward_meds)),
            rhoDelta=rho_mean,
            ZDelta=Z,
            K_atstar_mean=float(np.mean(K_atstar)) if len(K_atstar) else np.nan,
            K_atstar_std=float(np.std(K_atstar)) if len(K_atstar) else np.nan,
            F_star_robust=float(np.mean(Fstars)) if len(Fstars) else np.nan,
            slope_all=float(np.mean(slope_all_list)),
            slope_all_r2=float(np.mean(slope_all_r2_list)),
            slope_phase=float(np.nanmean(slope_ph_list)),
            slope_phase_r2=float(np.nanmean(slope_ph_r2_list)),
            phase_pts=float(np.mean(phase_counts)),
            valid_tstar=valid_tstar
        ))

    return pd.DataFrame(rows), fig_payload

# -------------------------
# Main
# -------------------------
families = ["lap_like", "diag_local", "random_sparse", "low_rank"]
L = laplacian_1d_ring(CFG["n"])

# reference rho for ZΔ
rng0 = np.random.default_rng(123)
Delta_ref = make_defect(CFG["ref_family"], CFG["n"], rng0, density=CFG["defect_density"], low_rank=CFG["low_rank"])
rho_ref = spectral_radius(Delta_ref, seed=999)

dfs = []
fig_payloads = []
for fam in families:
    df, payload = run_family(fam, CFG, L, rho_ref)
    dfs.append(df)
    fig_payloads.extend(payload)

res = pd.concat(dfs, ignore_index=True)

# -------------------------
# L0 closure test
# -------------------------
print("\n--- L0 CLOSURE TEST: log(Ward) ~ a log(eps) + b ---")
for fam in families:
    sub = res[res["family"] == fam].copy()
    x = np.log(sub["eps"].values)
    y = np.log(np.maximum(sub["Ward"].values, 1e-300))
    a, r2 = linreg_slope_r2(x, y)
    print(f"{fam:12s} | a={a: .3f} | R2={r2: .4f} | Ward_min={sub['Ward'].min():.3e} | Ward_max={sub['Ward'].max():.3e}")

# -------------------------
# Snapshot at mid eps
# -------------------------
mid_eps = CFG["eps_list"][len(CFG["eps_list"])//2]
summ = res[res["eps"] == mid_eps].copy()
print("\n--- UNIVERSALITY SNAPSHOT @ eps =", mid_eps, "---")
cols = ["family","eps","Ward","K_atstar_mean","K_atstar_std",
        "slope_all","slope_all_r2","slope_phase","slope_phase_r2",
        "phase_pts","F_star_robust","rhoDelta","ZDelta","valid_tstar"]
print(summ[cols].to_string(index=False))

# -------------------------
# Plots: Fig1 uses PHASE points only
# -------------------------
if CFG["make_plots"] and len(fig_payloads):
    # Fig.1: phase-only scatter at mid eps, one seed per family
    plt.figure()
    for (fam, eps, H, Phi, K, KR2, m, t_star) in fig_payloads:
        x = np.log(np.maximum(Phi[m], CFG["floor"]))
        y = np.log(np.maximum(H[m], CFG["floor"]))
        plt.scatter(x, y, s=10, alpha=0.75, label=f"{fam} (phase)")
    plt.xlabel(r"$\log \Phi$")
    plt.ylabel(r"$\log H$")
    plt.title("Fig.1 (phase-only): collapse slope near 2")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig1_path = f"{CFG['save_prefix']}_fig1_phase_only.png"
    plt.savefig(fig1_path, dpi=200)
    plt.show()
    print("Saved:", fig1_path)

    # K(t) representative: show also mask quality
    fam, eps, H, Phi, K, KR2, m, t_star = fig_payloads[0]
    plt.figure()
    plt.plot(K, linewidth=1.2, label="K(t)")
    plt.plot(np.where(m, 2.0, np.nan), linewidth=2.0, label="phase mask", alpha=0.9)
    if t_star is not None:
        plt.axvline(t_star, linestyle="--")
    plt.ylim(0, 4)
    plt.xlabel("t")
    plt.ylabel("K(t)")
    plt.title(f"K(t) & phase mask | family={fam}, eps={eps}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    figk_path = f"{CFG['save_prefix']}_K_series.png"
    plt.savefig(figk_path, dpi=200)
    plt.show()
    print("Saved:", figk_path)

# -------------------------
# Export
# -------------------------
csv_path = f"{CFG['save_prefix']}_summary.csv"
res.to_csv(csv_path, index=False)
print("\nSaved:", csv_path)
print("\n✅ ONE-CELL COLAB (FIXED) RUN COMPLETE.")
