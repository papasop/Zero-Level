# -*- coding: utf-8 -*-
"""
Closure → K≈2 → G 的论文级示例（无中文绘图 label 版）

流程：
1. 构造 1D 链式 Laplacian L
2. 构造有缺陷的 B = L + eps * Δ
3. 在 B-动力学下演化 u(t)，定义 collapse 场 χ = (L-B)u
4. 用 χ(t) 计算 H, Φ, Φ_c, K
5. 计算 Ward（closure 质量）和 SCI（时间相干性）
6. 在 eps0 上用 K≈2 通道 & φ² 一次性标定 |G|=G_CODATA
7. 固定 φ²_calib，扫 eps，得到 G(eps)、Ward(eps)，拟合幂律：
   |G|/G_CODATA ~ (Ward/Ward0)^(-α)
"""

import numpy as np
import matplotlib.pyplot as plt

# =============== 物理常数与单位 ===============
G_CODATA = 6.67430e-11  # SI
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# =============== 工具函数 ===============

def build_1d_chain_laplacian(N):
    E = N - 1
    D = np.zeros((E, N))
    for e in range(E):
        i, j = e, e+1
        D[e, i] = -1.0
        D[e, j] =  1.0
    w = np.ones(E)
    W_E = np.diag(w)
    L = D.T @ W_E @ D
    return D, W_E, L

def build_B_from_L(L, eps=1e-3, random_seed=0):
    np.random.seed(random_seed)
    N = L.shape[0]
    R = np.random.randn(N, N)
    R = 0.5 * (R + R.T)
    ones = np.ones((N,1))
    row_sum = R @ ones
    mean_corr = (row_sum @ ones.T) / N
    Delta = R - mean_corr
    B = L + eps * Delta
    return B

def simulate_u_with_B(B, T=400, dt=0.01, kappa=1.0, noise_level=0.1, random_seed=0):
    np.random.seed(random_seed)
    N = B.shape[0]
    u = np.random.randn(N)
    u_ts = [u.copy()]
    for _ in range(T-1):
        noise = noise_level * np.sqrt(dt) * np.random.randn(N)
        du = -kappa * (B @ u) + noise
        u = u + dt * du
        u_ts.append(u.copy())
    return np.array(u_ts)

def compute_chi_time_series(L, B, u_ts):
    Delta = L - B
    chi_ts = (Delta @ u_ts.T).T
    return chi_ts

def compute_H_Phi(chi_ts, dt):
    H_t   = np.sum(chi_ts**2, axis=1)
    Phi_t = np.sum(np.abs(chi_ts), axis=1)

    eps_reg = 1e-30
    logH   = np.log(H_t   + eps_reg)
    logPhi = np.log(Phi_t + eps_reg)

    dlogH_dt   = np.gradient(logH,   dt)
    dlogPhi_dt = np.gradient(logPhi, dt)

    dlogPhi_dt_safe = dlogPhi_dt.copy()
    dlogPhi_dt_safe[np.abs(dlogPhi_dt_safe) < 1e-20] = 1e-20
    K_t = dlogH_dt / dlogPhi_dt_safe

    Phi_c_t = np.gradient(Phi_t, dt)

    return dict(
        H_t=H_t,
        Phi_t=Phi_t,
        Phi_c_t=Phi_c_t,
        K_t=K_t
    )

def compute_Ward_and_SCI(L, B, u_ts, chi_ts):
    Bu_ts = (B @ u_ts.T).T

    norms_chi = np.linalg.norm(chi_ts, axis=1)
    norms_Bu  = np.linalg.norm(Bu_ts, axis=1) + 1e-20

    ratio = norms_chi / norms_Bu
    Ward = np.median(ratio)

    cos_list = []
    for t in range(1, len(chi_ts)):
        v1 = chi_ts[t-1]
        v2 = chi_ts[t]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-20 or n2 < 1e-20:
            continue
        cos = np.dot(v1, v2)/(n1*n2)
        cos_list.append(abs(cos))
    SCI = np.mean(cos_list) if len(cos_list)>0 else np.nan

    return dict(Ward=Ward, SCI=SCI)

def find_t_star_for_K(H_Phi, K_target=2.0, dt=0.01):
    H_t     = H_Phi["H_t"]
    Phi_c_t = H_Phi["Phi_c_t"]
    K_t     = H_Phi["K_t"]

    mask = (H_t > 1e-12) & (np.abs(Phi_c_t) > 1e-8)
    if not np.any(mask):
        return None, None, None

    idxs = np.where(mask)[0]
    K_diff = np.abs(K_t[mask] - K_target)
    best = np.argmin(K_diff)
    t_idx = idxs[best]
    t_star = t_idx * dt
    return t_idx, t_star, K_t[t_idx]

def compute_G2_from_chi(H_Phi, t_star_idx, phi2, L_L, L_T, L_M):
    H_t     = H_Phi["H_t"]
    Phi_c_t = H_Phi["Phi_c_t"]

    H_star     = H_t[t_star_idx]
    Phi_c_star = Phi_c_t[t_star_idx]

    denom = H_star * Phi_c_star
    if np.abs(denom) < 1e-30:
        return None, None, dict(valid=False)

    G_struct_signed = phi2 / denom
    G_phys_signed   = G_struct_signed * (L_L**3) / (L_M * (L_T**2))
    G_phys_abs      = np.abs(G_phys_signed)

    return G_phys_signed, G_phys_abs, dict(
        valid=True,
        H_star=H_star,
        Phi_c_star=Phi_c_star
    )

def run_single_eps(eps, phi2, N=32, T=400, dt=0.01,
                   kappa=1.0, noise_level=0.1,
                   K_target=2.0,
                   seed_B=1, seed_u=0):
    D, W_E, L = build_1d_chain_laplacian(N)
    B = build_B_from_L(L, eps=eps, random_seed=seed_B)
    u_ts   = simulate_u_with_B(B, T=T, dt=dt, kappa=kappa,
                               noise_level=noise_level,
                               random_seed=seed_u)
    chi_ts = compute_chi_time_series(L, B, u_ts)
    H_Phi  = compute_H_Phi(chi_ts, dt)
    closet = compute_Ward_and_SCI(L, B, u_ts, chi_ts)

    t_idx, t_star, K_star = find_t_star_for_K(H_Phi, K_target=K_target, dt=dt)
    if t_idx is None:
        return dict(
            eps=eps,
            Ward=closet["Ward"],
            SCI=closet["SCI"],
            valid=False,
            msg="no valid t*",
            G_phys=None,
            G_phys_abs=None
        )

    G_signed, G_abs, info_G = compute_G2_from_chi(
        H_Phi, t_idx, phi2, L_L=L_L, L_T=L_T, L_M=L_M
    )
    if not info_G["valid"]:
        return dict(
            eps=eps,
            Ward=closet["Ward"],
            SCI=closet["SCI"],
            valid=False,
            msg="denominator too small",
            G_phys=None,
            G_phys_abs=None
        )

    return dict(
        eps=eps,
        Ward=closet["Ward"],
        SCI=closet["SCI"],
        valid=True,
        t_star=t_star,
        K_star=K_star,
        G_phys=G_signed,
        G_phys_abs=G_abs,
        H_star=info_G["H_star"],
        Phi_c_star=info_G["Phi_c_star"]
    )

# =============== Step 1: 在 eps0 上标定 φ²（保证 φ² > 0） ===============

eps0 = 1e-3
phi2_init = 1.0

print("=== Step 1: 在 eps0 上用 G_CODATA 标定 φ²（取 |G| 对齐） ===")
res_base = run_single_eps(eps0, phi2_init)

if not res_base["valid"]:
    raise RuntimeError(f"eps0={eps0} 上没有得到有效 G: {res_base['msg']}")

Ward0    = res_base["Ward"]
G0_signed= res_base["G_phys"]
G0_abs   = res_base["G_phys_abs"]

phi2_calib = phi2_init * (G_CODATA / G0_abs)

print(f"eps0               = {eps0:.3e}")
print(f"Ward0              = {Ward0:.6e}")
print(f"G_phys_signed(eps0, φ²=1) = {G0_signed:.6e}  [SI]")
print(f"|G_phys|(eps0, φ²=1)       = {G0_abs:.6e}  [SI]")
print(f"G_CODATA                  = {G_CODATA:.6e}  [SI]")
print(f"phi2_calibrated           = {phi2_calib:.6e}")
print()

res_check = run_single_eps(eps0, phi2_calib)
G_check_abs = res_check["G_phys_abs"]
rel_err0 = abs(G_check_abs - G_CODATA)/G_CODATA*100

print("=== 用 φ²_calib 重跑 eps0 ===")
print(f"Ward(eps0)       = {res_check['Ward']:.6e}")
print(f"K(t*)            = {res_check['K_star']:.4f}")
print(f"t*               = {res_check['t_star']:.4f}")
print(f"G_phys_signed    = {res_check['G_phys']:.6e}")
print(f"|G_phys|(eps0)   = {G_check_abs:.6e}")
print(f"G_CODATA         = {G_CODATA:.6e}")
print(f"rel_err(|G|)     = {rel_err0:.4f} %")
print()

# =============== Step 2: 扫 eps，固定 φ²_calib ===============

eps_list = np.array([1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 2e-2, 5e-2])

results = []
print("=== Step 2: 扫 eps, 输出 Ward / SCI / |G|/G_CODATA ===")
for eps in eps_list:
    res = run_single_eps(eps, phi2_calib)
    if not res["valid"]:
        print(f"eps={eps:.3e}: 无有效 G ({res['msg']})")
        continue

    G_signed = res["G_phys"]
    G_abs    = res["G_phys_abs"]
    G_ratio  = G_abs / G_CODATA
    rel_err  = abs(G_abs - G_CODATA)/G_CODATA*100

    print(f"\n--- eps = {eps:.3e} ---")
    print(f"Ward            = {res['Ward']:.6e}")
    print(f"SCI             = {res['SCI']:.3f}")
    print(f"t* (K≈2)        = {res['t_star']:.4f}")
    print(f"K(t*)           = {res['K_star']:.4f}")
    print(f"G_phys_signed   = {G_signed:.6e}")
    print(f"|G_phys|(eps)   = {G_abs:.6e}")
    print(f"|G|/G_CODATA    = {G_ratio:.3e}")
    print(f"rel_err(|G|)    = {rel_err:.3f} %")

    results.append(dict(
        eps=eps,
        Ward=res["Ward"],
        SCI=res["SCI"],
        G_signed=G_signed,
        G_abs=G_abs,
        G_ratio=G_ratio
    ))

if len(results) == 0:
    raise RuntimeError("没有任何 eps 得到有效 G，检查参数设置。")

# =============== Step 3: 幂律拟合 |G|/G_CODATA vs Ward/Ward0 ===============

Ward_arr   = np.array([r["Ward"]    for r in results])
Gratio_arr = np.array([r["G_ratio"] for r in results])
Ward_rel   = Ward_arr / Ward0

mask = (Ward_rel > 0) & (Gratio_arr > 0)
logW = np.log(Ward_rel[mask])
logG = np.log(Gratio_arr[mask])

coeffs = np.polyfit(logW, logG, 1)
b, a = coeffs[0], coeffs[1]
alpha = -b

logG_pred = a + b * logW
SS_res = np.sum((logG - logG_pred)**2)
SS_tot = np.sum((logG - np.mean(logG))**2)
R2 = 1 - SS_res/SS_tot

print("\n=== Step 3: 幂律拟合结果（log–log 空间） ===")
print("拟合:  log(|G|/G_CODATA) ≈ a + b log(Ward/Ward0)")
print(f"a ≈ {a:.4f},  b ≈ {b:.4f}  → alpha = -b ≈ {alpha:.4f}")
print(f"R² (log–log)   ≈ {R2:.4f}")
print("\n用于拟合的数据点（Ward_rel, |G|/G_CODATA）:")
for w, g in zip(Ward_rel[mask], Gratio_arr[mask]):
    print(f"  {w:.3e},  {g:.3e}")

# =============== Step 4: 画图（K(t), H–Φ, 幂律 G–Ward） ===============

# 4.1 基准 eps0 的 K(t) & H–Φ 轨迹
D, W_E, L = build_1d_chain_laplacian(32)
B0 = build_B_from_L(L, eps=eps0, random_seed=1)
u_ts0 = simulate_u_with_B(B0, T=400, dt=0.01, kappa=1.0, noise_level=0.1, random_seed=0)
chi_ts0 = compute_chi_time_series(L, B0, u_ts0)
HP0 = compute_H_Phi(chi_ts0, dt=0.01)
t_axis = np.arange(len(HP0["H_t"])) * 0.01

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(t_axis, HP0["K_t"])
plt.axhline(2.0, color='gray', linestyle='--', label='K=2')
plt.xlabel("t")
plt.ylabel("K(t)")
plt.title(f"K(t) at eps0={eps0:.0e}")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1,2,2)
plt.loglog(HP0["Phi_t"]+1e-30, HP0["H_t"]+1e-30, '.-')
plt.xlabel(r"$\Phi(t)$")
plt.ylabel(r"$H(t)$")
plt.title("H–Phi scaling trajectory (log-log)")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.show()

# 4.2 |G|/G_CODATA vs Ward/Ward0 幂律图
plt.figure(figsize=(6,5))
plt.loglog(Ward_rel, Gratio_arr, "o", label="data")

W_min, W_max = Ward_rel.min(), Ward_rel.max()
W_plot = np.logspace(np.log10(W_min), np.log10(W_max), 200)
logW_plot = np.log(W_plot)
logG_fit = a + b * logW_plot
G_fit = np.exp(logG_fit)

plt.loglog(W_plot, G_fit, "-", label=f"fit: alpha≈{alpha:.2f}, R²≈{R2:.3f}")
plt.axvline(1.0, color="gray", linestyle="--", linewidth=1)
plt.axhline(1.0, color="gray", linestyle="--", linewidth=1)
plt.xlabel(r"$\mathrm{Ward} / \mathrm{Ward}_0$")
plt.ylabel(r"$|G| / G_{\mathrm{CODATA}}$")
plt.title(r"$|G|$ vs closure residual (Ward)" + f"\nalpha≈{alpha:.2f}, R²≈{R2:.3f}")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()
plt.show()
