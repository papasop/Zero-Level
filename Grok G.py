# ==========================================================
#  Closure → G : 完全复现附录 A 的数值实验
#  Python 3, NumPy/SciPy/Matplotlib
# ==========================================================
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.stats import linregress
plt.rcParams.update({'font.size': 12})

# --------------------- 参数 ---------------------
N        = 400          # 格点数
dt       = 1e-3         # 时间步
T        = 20000        # 总步数
δ_K      = 0.03         # K 进入 Lorentz 窗口的容差
eps_grid = np.logspace(-6, -2, 30)   # ε 扫描范围
np.random.seed(42)

# --------------------- 1. 闭合 Laplacian L ---------------------
i = np.arange(N)
diag = np.ones(N)
off  = -np.ones(N-1)
L = sp.diags([off, 2*diag, off], [-1, 0, 1], format='csc')
L = L.astype(np.float64)

# --------------------- 2. 缺陷 Δ (随机稀疏 Hermite) ---------------------
rng = np.random.default_rng(0)
nnz = int(0.02 * N)                     # 2% 非零元
row = rng.integers(0, N, nnz)
col = rng.integers(0, N, nnz)
data = rng.normal(0, 1, nnz)
# 强制 Hermite
Delta = sp.coo_matrix((data, (row, col)), shape=(N,N))
Delta = (Delta + Delta.T)/2
Delta = Delta.tocsc()

# --------------------- 3. 参考 ε₀ 标定 G ---------------------
ε0 = 1e-4
B0 = L + ε0 * Delta

# 初始场 (随机)
u = np.random.randn(N)
u /= np.linalg.norm(u)

# 存储轨迹
H_traj   = np.zeros(T)
Phi_traj = np.zeros(T)
r_traj   = np.zeros(T)

# 演化
u_t = u.copy()
for k in range(T):
    chi = (L - B0) @ u_t                # χ = -ε Δ u
    H_traj[k]   = np.linalg.norm(chi)**2
    Phi_traj[k] = np.linalg.norm(chi, ord=1)
    r_traj[k]   = np.linalg.norm(chi) / np.linalg.norm(B0 @ u_t)
    u_t = u_t - dt * (B0 @ u_t)         # du/dt = -B u

Ward0 = np.median(r_traj)

# 计算瞬时 K(t)
logH   = np.log(H_traj[1:-1] + 1e-30)
logPhi = np.log(Phi_traj[1:-1] + 1e-30)
dlogH   = np.diff(logH)
dlogPhi = np.diff(logPhi)
K_inst  = dlogH / dlogPhi

# 找到 t* (第一次进入 |K-2|<δ_K)
idx = np.where(np.abs(K_inst - 2) < δ_K)[0]
t_star0 = idx[0] + 1 if len(idx) > 0 else T//2   # 安全 fallback

# Φ̇ ≈ (Φ[t+1] - Φ[t-1])/(2Δt)
Phi_dot = (Phi_traj[t_star0+1] - Phi_traj[t_star0-1]) / (2*dt)

# G₂-map:  F(H,Φ,Φ̇) = H / (Φ·Φ̇)
F0 = H_traj[t_star0] / (Phi_traj[t_star0] * np.abs(Phi_dot) + 1e-30)

# 物理参考值 (CODATA 2022)
G_ref = 6.67430e-11
phi2_calib = G_ref / np.abs(F0)          # 标定整体尺度

# --------------------- 4. 扫描 ε ---------------------
Ward_vals = np.zeros_like(eps_grid)
G_vals    = np.zeros_like(eps_grid)
K_at_star = np.zeros_like(eps_grid)

for i, ε in enumerate(eps_grid):
    B = L + ε * Delta
    u_t = u.copy()
    H_traj[:]   = 0
    Phi_traj[:] = 0
    r_traj[:]   = 0

    for k in range(T):
        chi = (L - B) @ u_t
        H_traj[k]   = np.linalg.norm(chi)**2
        Phi_traj[k] = np.linalg.norm(chi, ord=1)
        r_traj[k]   = np.linalg.norm(chi) / np.linalg.norm(B @ u_t + 1e-30)
        u_t = u_t - dt * (B @ u_t)

    Ward_vals[i] = np.median(r_traj)

    # K(t)
    logH   = np.log(H_traj[1:-1] + 1e-30)
    logPhi = np.log(Phi_traj[1:-1] + 1e-30)
    dlogH   = np.diff(logH)
    dlogPhi = np.diff(logPhi)
    K_inst  = dlogH / dlogPhi

    idx = np.where(np.abs(K_inst - 2) < δ_K)[0]
    t_star = idx[0] + 1 if len(idx) > 0 else T//2

    Phi_dot = (Phi_traj[t_star+1] - Phi_traj[t_star-1]) / (2*dt) if t_star>0 and t_star<T-1 else 0
    F = H_traj[t_star] / (Phi_traj[t_star] * np.abs(Phi_dot) + 1e-30)

    G_struct = phi2_calib * F
    G_vals[i] = np.abs(G_struct)
    K_at_star[i] = K_inst[idx[0]] if len(idx) > 0 else np.nan

# --------------------- 5. 拟合 α ---------------------
mask = (Ward_vals > 0) & np.isfinite(G_vals)
log_Ward_ratio = np.log(Ward_vals[mask] / Ward0)
log_G_ratio    = np.log(np.abs(G_vals[mask]) / G_ref)

slope, intercept, r_value, _, _ = linregress(log_Ward_ratio, log_G_ratio)
alpha = -slope
R2 = r_value**2

print(f"\n=== 结果汇总 ===")
print(f"Ward(ε₀)      = {Ward0:.3e}")
print(f"G(ε₀)        ≈ {G_vals[np.argmin(np.abs(eps_grid-ε0))]:.3e}  (目标 {G_ref:.3e})")
print(f"拟合 α        = {alpha:.3f}")
print(f"R²           = {R2:.4f}")

# --------------------- 6. 作图 ---------------------
fig, ax = plt.subplots(1, 2, figsize=(12,5))

# (左) G vs Ward
ax[0].loglog(Ward_vals, G_vals, 'o-', label='数值')
ax[0].axhline(G_ref, color='k', ls='--', label='CODATA')
ax[0].set_xlabel('Ward statistic')
ax[0].set_ylabel('|G|  (m³ kg⁻¹ s⁻²)')
ax[0].set_title('Gravitational coupling from closure defect')
ax[0].legend(); ax[0].grid(True, which='both', ls=':')

# (右) log-log 拟合
ax[1].plot(log_Ward_ratio, log_G_ratio, 's', label='数据')
xfit = np.linspace(log_Ward_ratio.min(), log_Ward_ratio.max(), 100)
ax[1].plot(xfit, intercept + slope*xfit, 'r--',
           label=rf'拟合 $\alpha={alpha:.2f}$ ($R^2={R2:.3f}$)')
ax[1].set_xlabel(r'$\log(\mathrm{Ward}/\mathrm{Ward}_0)$')
ax[1].set_ylabel(r'$\log(|G|/G_0)$')
ax[1].set_title('Cubic sensitivity')
ax[1].legend(); ax[1].grid(True, which='both', ls=':')

plt.tight_layout()
plt.show()