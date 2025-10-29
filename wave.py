# =========================================
# [Truthful PSW • Paper Cell]
# 1) 正确相位：theta = log p mod 2π（无“均匀化”）
# 2) 直接用 L_raw（不做循环投影/平均）
# 3) 评估：R²、R²₀(非中心化)、cos²、τ、spacing、MAE
# 仅 print/plt.show()，不写文件
# =========================================

import numpy as np, matplotlib.pyplot as plt, requests
from numpy.linalg import eigh
from scipy.stats import linregress, kendalltau, pearsonr

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12,8)

# ---------- 基础 ----------
def first_primes(P=1200):
    n=P; Nmax=max(int(n*(np.log(n)+np.log(np.log(n))))+50, 2000)
    while True:
        sieve = np.ones(Nmax+1, dtype=bool); sieve[:2]=False
        for i in range(2, int(Nmax**0.5)+1):
            if sieve[i]: sieve[i*i::i]=False
        ps = np.flatnonzero(sieve)
        if len(ps)>=P: return ps[:P]
        Nmax *= 2

def theta_from_primes_corrected(primes):
    # 关键修复：θ = log p mod 2π （不再累计、不再补均值）
    return (np.log(primes.astype(float)) % (2*np.pi))

def theta_graph_laplacian(theta, B=8, sigma=0.12):
    N=len(theta); order=np.argsort(theta); th=theta[order]
    def angdist(a,b): d=np.abs(a-b); return np.minimum(d, 2*np.pi-d)
    W=np.zeros((N,N))
    for i in range(N):
        for off in range(1,B+1):
            j1=(i-off)%N; j2=(i+off)%N
            w1=np.exp(-0.5*(angdist(th[i],th[j1])/sigma)**2)
            w2=np.exp(-0.5*(angdist(th[i],th[j2])/sigma)**2)
            W[i,j1]=w1; W[j1,i]=w1; W[i,j2]=w2; W[j2,i]=w2
    d=W.sum(axis=1)
    L=np.diag(d)-W
    inv=np.empty_like(order); inv[order]=np.arange(N)
    return L[np.ix_(inv,inv)]

def get_zeta_zeros(n_zeros=100):
    try:
        url=f"https://www.lmfdb.org/api/zeros/?n={n_zeros}&fmt=csv"
        r=requests.get(url,timeout=6); r.raise_for_status()
        vals=[]
        for line in r.text.strip().split("\n")[1:]:
            p=line.split(","); 
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

def metrics_truth(y, x):
    # 含截距线性
    slope,intercept,r,p,_=linregress(x,y)
    yhat=slope*x+intercept
    r2 = 1 - np.sum((y-yhat)**2)/(np.sum((y-y.mean())**2)+1e-12)
    mae = np.mean(np.abs(y-yhat))
    # 过原点 R²：非中心化分母
    s0 = float(np.dot(x,y)/(np.dot(x,x)+1e-12))
    y0 = s0*x
    r2_0 = 1 - np.sum((y-y0)**2)/(np.sum(y**2)+1e-12)
    # 余弦 R²（尺度稳健）
    cos2 = (np.dot(x,y)**2)/((np.dot(x,x)+1e-12)*(np.dot(y,y)+1e-12))
    # 秩一致 & spacing
    tau, tau_p = kendalltau(y, x)
    sp, sp_p = pearsonr(np.diff(y), np.diff(x)) if len(x)>=3 else (np.nan, np.nan)
    return dict(slope=slope, intercept=intercept, r2=r2, r2_0=r2_0, cos2=cos2,
                tau=tau, tau_p=tau_p, spacing_r=sp, spacing_p=sp_p, mae=mae, yhat=yhat)

# ---------- 主流程（真实版） ----------
P=1200; N0=100; B=8; sigma=0.12
pr = first_primes(P)
theta = theta_from_primes_corrected(pr)          # ← 修复的相位
L_raw = theta_graph_laplacian(theta, B=B, sigma=sigma)  # ← 不再做 circulant 投影
w, _ = eigh(L_raw)                               # 对称实矩阵；O(N^3) 但 P=1200 可承受
w = np.sort(np.real(w))
f = w[1:]                                        # 去掉常数模
t = get_zeta_zeros(N0)
n = min(len(f), len(t))
f = f[:n]; t = t[:n]

# 线性评估
m = metrics_truth(f, t)

# ---------- 输出 ----------
print("\n[Truthful PSW] L_raw eigen vs Im(ρ) （theta=log p mod 2π，no circulant）")
print(f"  slope={m['slope']:.6f}, intercept={m['intercept']:.6f}")
print(f"  R²={m['r2']:.3f}, R²₀(非中心化)={m['r2_0']:.3f}, R²₀(cos)={m['cos2']:.3f}")
print(f"  τ={m['tau']:.3f} (p={m['tau_p']:.2g}), spacing={m['spacing_r']:.3f} (p={m['spacing_p']:.2g})")
print(f"  MAE={m['mae']:.3e}")

# ---------- 作图 ----------
fig, axs = plt.subplots(1,3, figsize=(13,4))

# (1) 散点 + 线性拟合
axs[0].scatter(t, f, s=16, alpha=0.7, label='data')
axs[0].plot(t, m['yhat'], lw=2, label=f"Linear R²={m['r2']:.3f}")
axs[0].set_xlabel('Im(ρ)'); axs[0].set_ylabel('eigen(L_raw)')
axs[0].set_title('ζ zeros vs eigen(L_raw)')
axs[0].legend(); axs[0].grid(True)

# (2) Δ-spacing 对齐
axs[1].scatter(np.diff(t), np.diff(f), s=16, alpha=0.7)
axs[1].set_xlabel('Δ Im(ρ)'); axs[1].set_ylabel('Δ eigen')
axs[1].set_title(f'spacing corr={m["spacing_r"]:.3f} (p={m["spacing_p"]:.2g})')
axs[1].grid(True)

# (3) 残差
res = np.abs(f - m['yhat'])
axs[2].plot(res, 'o-', alpha=0.8)
axs[2].set_title(f'|residuals|  (MAE={m["mae"]:.2e})'); axs[2].set_xlabel('k'); axs[2].grid(True)

plt.tight_layout(); plt.show()
