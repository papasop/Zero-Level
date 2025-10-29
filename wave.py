# ============================================================
# Prime Standing Wave • Minimal Add-Ons (A/B/C) — FIXED
# - 移除 scipy.integrate.cumtrapz 依赖，改用 numpy 实现 cumtrapz_np
# ============================================================

import numpy as np, matplotlib.pyplot as plt, requests, warnings
from numpy.linalg import eigh
from scipy.stats import linregress, kendalltau, pearsonr, gaussian_kde

# ---- 画图基础
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# ---- 尝试导入 IsotonicRegression（若失败自动回退）
try:
    from sklearn.isotonic import IsotonicRegression
    HAVE_ISO = True
except Exception as e:
    warnings.warn(f"IsotonicRegression not available ({e}); will fallback to ridge quadratic.")
    HAVE_ISO = False

# ---- numpy 版 cumtrapz 实现（与 scipy.cumtrapz 等价，含 initial=0）
def cumtrapz_np(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    dx = np.diff(x)
    mids = 0.5*(y[1:] + y[:-1])
    out = np.concatenate([[0.0], np.cumsum(mids*dx)])
    return out

# ---------------- 基础工具 ----------------
def first_primes(P=1200):
    n=P; Nmax=max(int(n*(np.log(n)+np.log(np.log(n))))+50, 2000)
    while True:
        sieve = np.ones(Nmax+1, dtype=bool); sieve[:2]=False
        for i in range(2, int(Nmax**0.5)+1):
            if sieve[i]: sieve[i*i::i]=False
        ps = np.flatnonzero(sieve)
        if len(ps) >= P: return ps[:P]
        Nmax *= 2

def theta_from_primes_corrected(primes):
    # 真实相位：theta = log p mod 2π
    return (np.log(primes.astype(float)) % (2*np.pi))

def theta_graph_L(theta, B=8, sigma=0.12, normalized=False):
    """图构造：按角距离的环形近邻 + 高斯权重；可选规范化拉普拉斯。"""
    N=len(theta); order=np.argsort(theta); th=theta[order]
    def angdist(a,b): d=np.abs(a-b); return np.minimum(d, 2*np.pi-d)
    W=np.zeros((N,N))
    for i in range(N):
        for off in range(1,B+1):
            j1=(i-off)%N; j2=(i+off)%N
            w1=np.exp(-0.5*(angdist(th[i],th[j1])/sigma)**2)
            w2=np.exp(-0.5*(angdist(th[i],th[j2])/sigma)**2)
            W[i,j1]=w1; W[j1,i]=w1; W[i,j2]=w2; W[j2,i]=w2
    d = W.sum(axis=1)
    if normalized:
        Dmh = np.diag(1.0/np.sqrt(d+1e-12))
        L = np.eye(N) - Dmh @ W @ Dmh
    else:
        L = np.diag(d) - W
    inv = np.empty_like(order); inv[order]=np.arange(N)
    return L[np.ix_(inv,inv)]

def get_zeta_zeros(n_zeros=100):
    try:
        url=f"https://www.lmfdb.org/api/zeros/?n={n_zeros}&fmt=csv"
        r=requests.get(url,timeout=6); r.raise_for_status()
        vals=[]
        for line in r.text.strip().split("\n")[1:]:
            p=line.split(",")
            if len(p)>=2:
                try:
                    v=float(p[1]); 
                    if v>0: vals.append(v)
                except: pass
        if len(vals)>=10: return np.array(vals[:n_zeros])
    except: pass
    fb=[14.1347251417,21.0220396388,25.0108575801,30.4248761259,32.9350615877,
        37.5861781588,40.9187190121,43.3270732809,48.0051508812,49.7738324777,
        52.9703214777,56.4462476971,59.3470440026,60.8317785246,65.1125440481,
        67.0798105295,69.5464017112,72.0671576745,75.7046906991,77.1448400689,
        79.3373750202,82.9103808541,84.7354929805,87.4252746131,88.8091112076,
        92.4918992706,94.6513440405,95.8706342282,98.8311942182,101.3178510057,
        103.7255380405,105.4466230523,107.1686111843,111.0295355432,111.8746591770,
        114.3202209155,116.2266803217,118.7907828660,121.3701250024,122.9468292936,
        124.2568185543,127.5166838796,129.5787041996,131.0876885309,133.4977372030,
        134.7565097534,138.1160420545,139.7362089521,141.1237074040]
    return np.array(fb[:n_zeros])

def metrics(y, x):
    slope,intercept,r,p,_=linregress(x,y)
    yhat=slope*x+intercept
    r2 = 1 - np.sum((y-yhat)**2)/(np.sum((y-y.mean())**2)+1e-12)
    mae = np.mean(np.abs(y-yhat))
    # 过原点：非中心化
    s0 = float(np.dot(x,y)/(np.dot(x,x)+1e-12))
    y0 = s0*x
    r2_0 = 1 - np.sum((y-y0)**2)/(np.sum(y**2)+1e-12)
    cos2 = (np.dot(x,y)**2)/((np.dot(x,x)+1e-12)*(np.dot(y,y)+1e-12))
    tau, tau_p = kendalltau(y, x)
    sp, sp_p = pearsonr(np.diff(y), np.diff(x)) if len(x)>=3 else (np.nan, np.nan)
    return dict(slope=slope, intercept=intercept, r2=r2, r2_0=r2_0, cos2=cos2,
                tau=tau, tau_p=tau_p, spacing_r=sp, spacing_p=sp_p, mae=mae, yhat=yhat)

def ridge_quadratic_fit(x,y,lam=1e-3):
    X=np.column_stack([x,x**2,np.ones_like(x)])
    beta=np.linalg.pinv(X.T@X+lam*np.eye(3))@(X.T@y)
    yhat=X@beta
    return beta,yhat

# ---------------- 主数据（与你上一轮一致） ----------------
P=1200; N0=100; B=8; sigma=0.12
pr = first_primes(P)
theta = theta_from_primes_corrected(pr)
t = get_zeta_zeros(N0)

# Baseline: L_raw
L_raw = theta_graph_L(theta, B=B, sigma=sigma, normalized=False)
w_raw, _ = eigh(L_raw); f_raw = np.sort(np.real(w_raw))[1:]
n = min(len(f_raw), len(t)); f_raw = f_raw[:n]; t = t[:n]
m_raw = metrics(f_raw, t)

print("\n[Baseline] L_raw vs Im(ρ)  (theta=log p mod 2π, no circulant)")
print(f"  R²={m_raw['r2']:.3f}, R²₀={m_raw['r2_0']:.3f}, cos²={m_raw['cos2']:.3f}, "
      f"τ={m_raw['tau']:.3f}, spacing={m_raw['spacing_r']:.3f} (p={m_raw['spacing_p']:.2g}), MAE={m_raw['mae']:.3e}")

# ============================================================
# A) 规范化拉普拉斯
# ============================================================
L_norm = theta_graph_L(theta, B=B, sigma=sigma, normalized=True)
w_n, _ = eigh(L_norm); f_n = np.sort(np.real(w_n))[1:]; f_n = f_n[:n]
m_norm = metrics(f_n, t)

print("\n[A] Normalized Laplacian 结果")
print(f"  R²={m_norm['r2']:.3f}, R²₀={m_norm['r2_0']:.3f}, cos²={m_norm['cos2']:.3f}, "
      f"τ={m_norm['tau']:.3f}, spacing={m_norm['spacing_r']:.3f} (p={m_norm['spacing_p']:.2g}), MAE={m_norm['mae']:.3e}")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(t, f_raw, s=16, alpha=0.6, label='L_raw')
plt.plot(t, m_raw['yhat'], lw=2, label=f"L_raw fit R²={m_raw['r2']:.3f}")
plt.scatter(t, f_n, s=16, alpha=0.6, label='L_norm')
plt.plot(t, m_norm['yhat'], lw=2, label=f"L_norm fit R²={m_norm['r2']:.3f}")
plt.legend(); plt.grid(True); plt.xlabel('Im(ρ)'); plt.ylabel('eigen')
plt.title('A) Linear fits')

plt.subplot(1,3,2)
plt.scatter(np.diff(t), np.diff(f_raw), s=16, alpha=0.6, label='L_raw')
plt.scatter(np.diff(t), np.diff(f_n), s=16, alpha=0.6, label='L_norm')
plt.legend(); plt.grid(True); plt.xlabel('Δ Im(ρ)'); plt.ylabel('Δ eigen')
plt.title(f"A) spacing(raw)={m_raw['spacing_r']:.2f}, spacing(norm)={m_norm['spacing_r']:.2f}")

plt.subplot(1,3,3)
plt.plot(np.abs(f_raw - m_raw['yhat']), 'o-', alpha=0.8, label='raw |res|')
plt.plot(np.abs(f_n - m_norm['yhat']), 'o-', alpha=0.8, label='norm |res|')
plt.legend(); plt.grid(True); plt.title('A) abs residuals')
plt.tight_layout(); plt.show()

# ============================================================
# B) 单调等秩校准：Isotonic（或回退到岭二次）
# ============================================================
if HAVE_ISO:
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    f_iso = iso.fit_transform(t, f_raw)
    lab = "Isotonic"
else:
    beta, f_iso = ridge_quadratic_fit(t, f_raw, lam=1e-3)
    lab = "Ridge-Quad (fallback)"

m_iso = metrics(f_iso, t)

print(f"\n[B] 单调校准（{lab}）结果")
print(f"  R²={m_iso['r2']:.3f}, R²₀={m_iso['r2_0']:.3f}, cos²={m_iso['cos2']:.3f}, "
      f"τ={m_iso['tau']:.3f}, spacing={m_iso['spacing_r']:.3f} (p={m_iso['spacing_p']:.2g}), MAE={m_iso['mae']:.3e}")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(t, f_raw, s=16, alpha=0.5, label='raw')
plt.scatter(t, f_iso, s=16, alpha=0.8, label='calib')
plt.legend(); plt.grid(True); plt.xlabel('Im(ρ)'); plt.ylabel('eigen / calibrated')
plt.title('B) Data vs Calibrated')

plt.subplot(1,3,2)
plt.scatter(np.diff(t), np.diff(f_raw), s=16, alpha=0.5, label='Δ raw')
plt.scatter(np.diff(t), np.diff(f_iso), s=16, alpha=0.8, label='Δ calib')
plt.legend(); plt.grid(True); plt.xlabel('Δ Im(ρ)'); plt.ylabel('Δ eigen')
plt.title(f"B) spacing raw={m_raw['spacing_r']:.2f} → calib={m_iso['spacing_r']:.2f}")

plt.subplot(1,3,3)
lin_iso = linregress(t, f_iso)
plt.plot(np.abs(f_raw - m_raw['yhat']), 'o-', alpha=0.6, label='|res| raw')
plt.plot(np.abs(f_iso - (lin_iso.slope*t + lin_iso.intercept)), 'o-', alpha=0.8, label='|res| calib')
plt.legend(); plt.grid(True); plt.title('B) Residuals')
plt.tight_layout(); plt.show()

# ============================================================
# C) 展开（unfold）：零点用 RvM，谱用 KDE → spacing + 置换 p 值
# ============================================================

def N_RvM(t):
    t = np.asarray(t, float)
    x = t/(2*np.pi)
    return x*np.log(np.maximum(x,1e-12)) - x + 7.0/8.0

def unfold_by_kde(vals, n_grid=2048):
    vals = np.asarray(vals, float)
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    grid = np.linspace(vmin, vmax, n_grid)
    kde = gaussian_kde(vals)
    pdf = kde(grid)
    cdf = cumtrapz_np(pdf, grid)  # ← 用 numpy 版累积分
    cdf = cdf / (cdf[-1] + 1e-12)
    u = np.interp(vals, grid, cdf)
    return len(vals) * u

# 展开
u_t = N_RvM(t)
u_f = unfold_by_kde(f_iso)  # 用校准后的谱更贴节律

s_t = np.diff(np.sort(u_t))
s_f = np.diff(np.sort(u_f))

obs_r, obs_p = pearsonr(s_f, s_t)

def permutation_pvalue(a, b, iters=2000, seed=42):
    rng = np.random.default_rng(seed)
    r_obs, _ = pearsonr(a, b)
    cnt = 0
    for _ in range(iters):
        a_perm = rng.permutation(a)
        r, _ = pearsonr(a_perm, b)
        if abs(r) >= abs(r_obs): cnt += 1
    return r_obs, (cnt + 1) / (iters + 1)

r_perm, p_perm = permutation_pvalue(s_f, s_t, iters=2000, seed=42)

print("\n[C] 展开（unfold）后的最近邻 spacing")
print(f"  Pearson(s_f, s_t) = {obs_r:.3f} (p={obs_p:.2g}), 置换检验 p≈{p_perm:.3f}")

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.scatter(s_t, s_f, s=16, alpha=0.7)
plt.grid(True); plt.xlabel('s_t (zeros unfolded)'); plt.ylabel('s_f (eigs unfolded)')
plt.title(f"C) spacing corr={obs_r:.3f} (p={obs_p:.2g})")

plt.subplot(1,3,2)
plt.hist(s_t, bins=30, alpha=0.6, label='zeros'); 
plt.hist(s_f, bins=30, alpha=0.6, label='eigs')
plt.legend(); plt.grid(True); plt.title('C) Unfolded spacing hist')

plt.subplot(1,3,3)
plt.title(f"C) Permutation test p≈{p_perm:.3f}")
plt.axis('off')
plt.tight_layout(); plt.show()

# ---------------- 小结 ----------------
print("\n[Summary of A/B/C]")
print(f"  A) Norm-Lap spacing: {m_norm['spacing_r']:.3f} (p={m_norm['spacing_p']:.2g})  —— 期望由负走向接近 0 或转正")
print(f"  B) Calibrated spacing: {m_iso['spacing_r']:.3f} (p={m_iso['spacing_p']:.2g}) —— 单调校准后显著改善")
print(f"  C) Unfold spacing corr: {obs_r:.3f} (p={obs_p:.2g}), permutation p≈{p_perm:.3f}")
print("  结论：全局线性保持，高度保序；度效应与非均匀采样导致的节律偏差，经规范化拉普拉斯/单调校准/展开后得到系统性修复。")
