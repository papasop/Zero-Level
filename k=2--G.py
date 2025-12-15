# -*- coding: utf-8 -*-
"""
P4/P5 最终稳定运行版：修复了所有底层 Python 语法错误，并减小 dt=0.001 解决动力学不稳定性。
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import itertools

# =============== 物理常数与单位 (UNCHANGED) ===============
G_CODATA = 6.67430e-11
L_L = 1e-15
L_T = 1e-20
L_M = 0.063826

# =============== I. 核心工具函数 (修复所有 SyntaxError) ===============

def build_laplacian(N, dim=1):
    """构建 1D, 2D, 或 3D 方格图的 Laplacian L (使用标准 stencil)"""
    N_nodes = N**dim
    
    if dim == 1:
        D = np.zeros((N - 1, N))
        for i in range(N - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        L = D.T @ D
        return L, N_nodes
        
    elif dim == 2:
        L = np.zeros((N_nodes, N_nodes))
        N_side = N
        for i in range(N_nodes):
            row, col = i // N_side, i % N_side
            neighbors = []
            
            # --- 最终修复点：彻底移除 2D 的单行分号 ---
            if col < N_side - 1:
                neighbors.append(i + 1)
            if row < N_side - 1:
                neighbors.append(i + N_side)
            if col > 0:
                neighbors.append(i - 1)
            if row > 0:
                neighbors.append(i - N_side)
            # ----------------------------------------
                
            L[i, i] = len(neighbors)
            for j in neighbors: L[i, j] = -1.0
        return L, N_nodes
    
    elif dim == 3:
        L = np.zeros((N_nodes, N_nodes))
        N_side = N
        N_plane = N * N
        
        for i in range(N_nodes):
            x = i % N_side
            y = (i // N_side) % N_side
            z = i // N_plane

            neighbors = []
            
            # --- 确保 3D 也是标准多行 ---
            if x < N_side - 1: neighbors.append(i + 1)
            if x > 0: neighbors.append(i - 1)
            if y < N_side - 1: neighbors.append(i + N_side)
            if y > 0: neighbors.append(i - N_side)
            if z < N_side - 1: neighbors.append(i + N_plane)
            if z > 0: neighbors.append(i - N_plane)
            # ---------------------------
            
            L[i, i] = len(neighbors)
            for j in neighbors: L[i, j] = -1.0
        return L, N_nodes
        
    else:
        raise NotImplementedError("Only 1D, 2D, and 3D implemented for this test.")

def build_4th_order_laplacian(N, dim):
    """P4: 构造 4th-order Laplacian"""
    N_nodes = N**dim
    L = np.zeros((N_nodes, N_nodes))
    N_side = N
    
    if N < 5: 
        return build_laplacian(N, dim)[0]

    C1 = 16.0 / 12.0
    C2 = -1.0 / 12.0
    
    for i in range(N_nodes):
        row_sum = 0.0
        
        for d in range(int(dim)):
            offset_1 = N_side**d
            offset_2 = 2 * offset_1
            
            for sign in [1, -1]:
                j1 = i + sign * offset_1
                if 0 <= j1 < N_nodes:
                    L[i, j1] += C1
                    row_sum += C1
                
                j2 = i + sign * offset_2
                if 0 <= j2 < N_nodes:
                    L[i, j2] += C2
                    row_sum += C2

        L[i, i] = -row_sum
        
    return L

def build_Delta(L, def_type, random_seed=1):
    """构造不同类型的缺陷算子 Delta"""
    np.random.seed(random_seed)
    N_nodes = L.shape[0]
    N = int(np.round(N_nodes**(1/3))) if int(np.round(N_nodes**(1/3)))**3 == N_nodes else (int(np.round(N_nodes**0.5)) if int(np.round(N_nodes**0.5))**2 == N_nodes else N_nodes)
    dim = int(np.round(np.log(N_nodes) / np.log(N))) if N > 1 else 1

    if def_type == 'Lap_Like':
        L_temp, _ = build_laplacian(N, dim=dim) 
        Noise = np.random.randn(N_nodes, N_nodes) * 0.05
        Noise = 0.5 * (Noise + Noise.T)
        Delta = L_temp + Noise
        Delta = Delta - Delta.mean()
        
    elif def_type == 'Diag_Local':
        Delta = np.zeros((N_nodes, N_nodes))
        num_defects = int(N_nodes * 0.1)
        defect_indices = np.random.choice(N_nodes, size=num_defects, replace=False)
        for i in defect_indices:
            Delta[i, i] = np.random.randn()
        Delta = Delta - np.diag(Delta).sum() / N_nodes * np.eye(N_nodes)

    elif def_type == 'Random_Sparse':
        R = np.random.randn(N_nodes, N_nodes)
        R = 0.5 * (R + R.T)
        ones = np.ones((N_nodes, 1))
        row_sum = R @ ones
        mean_corr = (row_sum @ ones.T) / N_nodes
        Delta = R - mean_corr
        
    elif def_type == 'FD_Laplacian':
        L_def, _ = build_laplacian(N, dim=dim) 
        Delta = L_def + np.diag(np.ones(N_nodes) * 0.01)
        Delta = Delta - Delta.mean()
        
    elif def_type == '4th_Order_FD_Laplacian':
        L_def = build_4th_order_laplacian(N, dim=dim)
        Delta = L_def + np.diag(np.ones(N_nodes) * 0.01)
        Delta = Delta - Delta.mean()
        
    return Delta

# --- 核心运行和分析函数 ---
def build_B_from_L(L, eps, def_type, random_seed):
    Delta = build_Delta(L, def_type, random_seed)
    return L + eps * Delta
def compute_H_Phi(chi_ts, dt):
    H_t = np.sum(chi_ts**2, axis=1); Phi_t = np.sum(np.abs(chi_ts), axis=1); eps_reg = 1e-30
    logH = np.log(H_t + eps_reg); logPhi = np.log(Phi_t + eps_reg)
    dlogH_dt = np.gradient(logH, dt); dlogPhi_dt = np.gradient(logPhi, dt)
    dlogPhi_dt_safe = dlogPhi_dt.copy(); dlogPhi_dt_safe[np.abs(dlogPhi_dt_safe) < 1e-15] = 1e-15 
    K_t = dlogH_dt / dlogPhi_dt_safe; Phi_c_t = np.gradient(Phi_t, dt)
    return dict(H_t=H_t, Phi_t=Phi_t, Phi_c_t=Phi_c_t, K_t=K_t)
def find_t_star_for_K(H_Phi, K_target=2.0, dt=0.01):
    H_t = H_Phi["H_t"]; Phi_c_t = H_Phi["Phi_c_t"]; K_t = H_Phi["K_t"]
    mask = (H_t > 1e-8) & (np.abs(Phi_c_t) > 1e-6) & (np.abs(K_t) < 5) 
    if not np.any(mask): return None, None, None
    idxs = np.where(mask)[0]; K_diff = np.abs(K_t[mask] - K_target); best = np.argmin(K_diff)
    t_idx = idxs[best]; t_star = t_idx * dt
    return t_idx, t_star, K_t[t_idx]
def compute_G2_from_chi(H_Phi, t_star_idx, phi2):
    H_star = H_Phi["H_t"][t_star_idx]; Phi_star = H_Phi["Phi_t"][t_star_idx]; Phi_c_star = H_Phi["Phi_c_t"][t_star_idx]
    denom = Phi_star * np.abs(Phi_c_star)
    if denom < 1e-18: return None, None, dict(valid=False, F_star=np.nan)
    F_star = H_star / denom
    G_phys_abs = np.abs(phi2 * F_star) * (L_L**3) / (L_M * (L_T**2))
    return G_phys_abs, G_phys_abs, dict(valid=True, F_star=F_star)
def compute_Ward(L, B, u_ts, chi_ts):
    Bu_ts = (B @ u_ts.T).T; norms_chi = np.linalg.norm(chi_ts, axis=1); norms_Bu = np.linalg.norm(Bu_ts, axis=1) + 1e-20
    ratio = norms_chi / norms_Bu
    return np.median(ratio)
def simulate_u_with_B(B, T=400, dt=0.01, kappa=1.0, noise_level=0.1, random_seed=0):
    np.random.seed(random_seed); N = B.shape[0]; u = np.random.randn(N); u_ts = [u.copy()]
    for _ in range(T - 1):
        noise = noise_level * np.sqrt(dt) * np.random.randn(N)
        du = -kappa * (B @ u) + noise
        u = u + dt * du
        u_ts.append(u.copy())
    return np.array(u_ts)
def compute_chi_time_series(L, B, u_ts):
    Delta = L - B
    return (Delta @ u_ts.T).T

def run_single_eps_full(N, dim, L, eps, phi2, def_type, **kwargs):
    dt = kwargs.get('dt', 0.001); 
    B = build_B_from_L(L, eps=eps, def_type=def_type, random_seed=kwargs.get('seed_B', 1)); u_ts = simulate_u_with_B(B, dt=dt, random_seed=kwargs.get('seed_u', 0)); chi_ts = compute_chi_time_series(L, B, u_ts); H_Phi = compute_H_Phi(chi_ts, dt=dt); Ward = compute_Ward(L, B, u_ts, chi_ts)
    t_idx, t_star, K_star = find_t_star_for_K(H_Phi, dt=dt); 
    if t_idx is None: return dict(valid=False)
    G_abs, G_ratio, info_G = compute_G2_from_chi(H_Phi, t_idx, phi2); 
    if not info_G["valid"]: return dict(valid=False)
    K_valid = H_Phi['K_t'][(H_Phi['K_t'] != 0) & (np.abs(H_Phi['K_t']) < 5)] 
    H_t, Phi_t, Phi_c_t = H_Phi["H_t"], H_Phi["Phi_t"], H_Phi["Phi_c_t"]
    steady_mask = (H_t > 1e-8) & (np.abs(H_Phi['K_t']) < 5) & (Phi_t > 1e-8) & (np.abs(Phi_c_t) > 1e-6)
    F_t_raw = H_t / (Phi_t * np.abs(Phi_c_t)); F_mean_steady = np.nanmean(F_t_raw[steady_mask])
    return dict(eps=eps, Ward=Ward, valid=True, K_star=K_star, G_phys_abs=G_abs, G_ratio=G_abs / G_CODATA, K_mean=np.nanmean(K_valid) if len(K_valid)>0 else np.nan, K_std=np.nanstd(K_valid) if len(K_valid)>0 else np.nan, N=N, dim=dim, def_type=def_type, F_star_tstar=info_G['F_star'], F_mean_steady=F_mean_steady if not np.isnan(F_mean_steady) else 0.0, N_nodes=L.shape[0])

def run_multi_eps_scan(N, dim, def_type, eps_list, eps0, **kwargs):
    L, N_nodes = build_laplacian(N, dim); phi2_init = 1.0
    res_base = run_single_eps_full(N=N, dim=dim, L=L, eps=eps0, phi2=phi2_init, def_type=def_type, **kwargs)
    if not res_base or not res_base.get("valid"): return None, None
    phi2_calib = phi2_init * (G_CODATA / res_base["G_phys_abs"]); G_calib_abs = res_base["G_phys_abs"]
    results_data = []
    for eps in eps_list:
        res = run_single_eps_full(N=N, dim=dim, L=L, eps=eps, phi2=phi2_calib, def_type=def_type, **kwargs)
        if res and res.get("valid"): results_data.append(res)
    if len(results_data) < 2: return None, None
    Ward_arr = np.array([r["Ward"] for r in results_data]); Gratio_arr = np.array([r["G_ratio"] for r in results_data]); Ward_rel = Ward_arr / res_base["Ward"]
    logW = np.log(Ward_rel); logG = np.log(Gratio_arr); slope, intercept, r_value, _, _ = linregress(logW, logG); alpha = -slope; R2 = r_value**2
    K_means = np.array([r["K_mean"] for r in results_data]); K_stds = np.array([r["K_std"] for r in results_data]); F_means = np.array([r["F_mean_steady"] for r in results_data])
    return dict(def_type=def_type, dim=dim, N_nodes=N_nodes, N=N, alpha=alpha, R2=R2, data_count=len(results_data), K_mean_overall=np.nanmean(K_means), K_std_mean=np.nanmean(K_stds), F_mean_overall=np.nanmean(F_means), Ward0=res_base["Ward"], phi2_calib=phi2_calib, G_calib_abs=G_calib_abs), results_data


# =============== II. 主程序：测试执行 (P4/P5 实施) ===============

# --- 实验设置 ---
N_side_1D_base = 32
N_side_2D_base = 8 
N_side_3D_base = 6
eps0 = 1e-3

# 1. 普适性测试范围 (正常范围)
eps_scan_normal = np.logspace(np.log10(1e-4), np.log10(5e-2), 7) 
# 2. 极低 epsilon 收敛性范围 (P5: 极限测试)
eps_scan_micro = np.logspace(-16, -8, 10) 

# --- 家族定义 (P4: 新增 4th Order Laplacian, 包含 3D) ---
test_families = [
    dict(name="1D Lap Like (R2-Base)", def_type="Lap_Like", dim=1, N=N_side_1D_base), 
    dict(name="2D FD Laplacian (N=64)", def_type="FD_Laplacian", dim=2, N=N_side_2D_base), 
    dict(name="3D FD Laplacian (N=216)", def_type="FD_Laplacian", dim=3, N=N_side_3D_base),
    dict(name="2D 4th-Order FD Lap", def_type="4th_Order_FD_Laplacian", dim=2, N=N_side_2D_base),
    dict(name="2D Random Sparse (K-Base)", def_type="Random_Sparse", dim=2, N=N_side_2D_base), 
]

all_summary = []
all_raw_data = {}
print("--- STARTING ENHANCED MULTI-FAMILY CLOSURE TEST (P4/P5: Higher Order & Micro Epsilon) ---")

# --- I. 多族普适性测试 (正常范围: 1D, 2D, 3D, 4th Order) ---
print("\n[I. Universality Test (Normal Epsilon Range)]")
for family in test_families:
    summary, raw_data = run_multi_eps_scan(
        N=family['N'], dim=family['dim'], def_type=family['def_type'], 
        eps_list=eps_scan_normal, eps0=eps0
    )
    if summary:
        all_summary.append(summary)
        all_raw_data[family['name']] = raw_data
        print_str = f"PASS: {family['name']:<25} | Dim={summary['dim']} | N_nodes={summary['N_nodes']} | R2={summary['R2']:.4f} | K_mean={summary['K_mean_overall']:.4f} | F_mean={summary['F_mean_overall']:.3e}"
        print(print_str)
    else:
        print(f"FAIL: {family['name']:<25} | Not enough valid data points generated.")

# --- II. 微小 ε 极限收敛测试 (P5: 极限测试) ---
print("\n[II. Micro Epsilon Convergence Test (ε → 1e-16)]")
try:
    op_family = next(f for f in test_families if f['def_type'] == "Lap_Like" and f['dim']==1) 
except StopIteration:
    op_family = None

if op_family and any(s['def_type'] == op_family['def_type'] for s in all_summary):
    summary_micro, raw_data_micro = run_multi_eps_scan(
        N=op_family['N'], dim=op_family['dim'], def_type=op_family['def_type'], 
        eps_list=eps_scan_micro, eps0=eps0
    )
    if summary_micro:
        F_means = np.array([r['F_mean_steady'] for r in raw_data_micro])
        K_means = np.array([r['K_mean'] for r in raw_data_micro])
        Eps_list = np.array([r['eps'] for r in raw_data_micro])
        
        all_raw_data['Micro_Convergence'] = raw_data_micro
        
        print(f"PASS: Micro Convergence ({op_family['name']})")
        print(f"  Range: eps={Eps_list.min():.1e} to {Eps_list.max():.1e}")
        print(f"  Final K_mean={summary_micro['K_mean_overall']:.4f}")
        
        print(f"  Epsilon (limits): {Eps_list[[0, -1]]}")
        print(f"  F* Mean (limits): {F_means[[0, -1]]}")
        print(f"  K Mean (limits): {K_means[[0, -1]]}")
    else:
        print(f"FAIL: Micro Convergence test failed for {op_family['name']}.")
else:
    print("Skipping Micro Epsilon Test: Base family data not successfully generated in Stage I.")

# --- III. 跨家族预测测试 (F* Universality Check) ---
print("\n[III. Cross-Family Prediction Test (F* Universality Check)]")
try:
    # Safely retrieve reference data
    ref_summary = next(s for s in all_summary if s['def_type'] == 'Lap_Like' and s['dim']==1 and s['N_nodes']==N_side_1D_base)
    phi2_ref = ref_summary['phi2_calib']
    G_calib_ref = ref_summary['G_calib_abs']
    F_ref = ref_summary['F_mean_overall']
except StopIteration:
    print("Skipping Cross-Family Prediction: Base family data not available.")
    F_ref = None

if F_ref:
    print(f"Reference F* (1D Lap Like Base, N={N_side_1D_base}): {F_ref:.3e}")
    print("-" * 60)

    for summary in all_summary: # Use summary directly
        L, _ = build_laplacian(summary['N'], summary['dim'])
        res_pred = run_single_eps_full(
            N=summary['N'], dim=summary['dim'], L=L, eps=eps0, phi2=phi2_ref, def_type=summary['def_type']
        )
        
        if res_pred and res_pred.get("valid"):
            G_abs_pred = res_pred['G_phys_abs']
            error = np.abs(G_abs_pred - G_calib_ref) / G_calib_ref * 100 
            
            F_mean_val = res_pred['F_mean_steady']
            F_ratio = F_mean_val / F_ref
            
            # --- KeyError Fix: Use summary['def_type'] and summary['N_nodes'] directly ---
            name_str = f"{summary['def_type']} (N={summary['N_nodes']})"
            print(f"  {name_str:<25} | N={summary['N_nodes']:<4}: G/G_ref={G_abs_pred/G_calib_ref:.4f} (Error={error:.2f}%)")
            print(f"    F* Ratio (F*/F_ref): {F_ratio:.4f} | F* (Target): {F_mean_val:.3e}")
            # -----------------------------------------------------------------------------
        else:
            print(f"  Prediction failed for {summary['def_type']} (N={summary['N_nodes']}).")


# =============== IV. 结果总结表格 (P4/P5) ===============
print("\n\n--- FINAL SUMMARY TABLE (P4/P5) ---")

print("="*120)
header = f"{'Defect Family':<25} | {'Dim':<5} | {'N_Nodes':<10} | {'Alpha':<10} | {'R-Squared' :<11} | {'K_mean':<10} | {'F_mean (Ref)':<15} | {'Conclusion'}"
print(header)
print("="*120)

F_ref_val = F_ref if F_ref is not None else 1.0

for res in all_summary:
    conclusion = "Order Parameter (Stable)" if res['R2'] < 0.2 else "Power Law (Sensitive)"
    F_ratio_print = res['F_mean_overall'] / F_ref_val
    print(f"{res['def_type']:<25} | {res['dim']:<5} | {res['N_nodes']:<10} | {res['alpha']:.4f}{'':<6} | {res['R2']:.4f}{'':<7} | {res['K_mean_overall']:.4f}{'':<10} | {F_ratio_print:.4f}{'':<15} | {conclusion}")
    
print("="*120)
